"""Agent module"""
from typing import Dict, List, Optional, Tuple, Any
import logging
import re
from collections import deque
import json
import os
import pandas as pd
from datetime import datetime
from workflow.data_processor import DataProcessor

# from workflow.prompts import *
from workflow.prompts import *


logger = logging.getLogger(__name__)

class GazeMotionAgent:
    """Gaze-motion agent"""
    
    def __init__(
        self,
        gaze_llm: Any,
        gaze_refinement_llm: Any,
        integrated_llm: Any,
        integrated_refinement_llm: Any,
        gaze_evaluation_llm: Any,
        integrated_evaluation_llm: Any,
        data_processor: DataProcessor,
        memory_use_count: int = 0,
        memory_buffer_size: int = 10,
        gaze_evaluation_threshold: float = 4.0,
        integrated_evaluation_threshold: float = 3.5,
        max_retry: int = 2,
        batch_size: int = 1,
    ):
        """
        Initialize agent
        
        Args:
            gaze_llm: LLM for generating gaze description
            gaze_refinement_llm: LLM for refining gaze description
            integrated_llm: LLM for generating integrated description
            integrated_refinement_llm: LLM for refining integrated description
            gaze_evaluation_llm: LLM for evaluating gaze description
            integrated_evaluation_llm: LLM for evaluating integrated description
            data_processor: Data processor
            memory_use_count: Number of historical memories to use, 0 means no history memory
            memory_buffer_size: Memory buffer size, default is 10
            gaze_evaluation_threshold: Gaze description evaluation threshold, below which the description will be regenerated
            integrated_evaluation_threshold: Integrated description evaluation threshold, below which the description will be regenerated
            max_retry: Maximum number of retries
            batch_size: Number of gaze descriptions to generate in batch, only valid for local models
        """
        self.gaze_llm = gaze_llm
        self.gaze_refinement_llm = gaze_refinement_llm
        self.integrated_llm = integrated_llm
        self.integrated_refinement_llm = integrated_refinement_llm
        self.gaze_evaluation_llm = gaze_evaluation_llm
        self.integrated_evaluation_llm = integrated_evaluation_llm
        self.data_processor = data_processor
        
        # Ensure memory_use_count is not greater than memory_buffer_size
        self.memory_buffer_size = memory_buffer_size
        self.memory_use_count = min(memory_use_count, memory_buffer_size)
        
        # Evaluation and refinement parameters
        self.gaze_evaluation_threshold = gaze_evaluation_threshold
        self.integrated_evaluation_threshold = integrated_evaluation_threshold
        self.max_retry = max_retry
        self.batch_size = max(1, batch_size)
        
        # Initialize memory buffer
        self.memory_buffer = deque(maxlen=self.memory_buffer_size)
        self.gaze_descriptions_cache = {}
    
    def _create_simplified_events_json(self, gaze_events: List[Dict]) -> str:
        """Create simplified gaze events JSON string"""
        simplified_events = []
        for event in gaze_events:
            simplified_event = {
                "event_type": event["event_type"],
                "duration_label": event["duration_label"]
            }
            
            # Add corresponding direction/position information based on event type
            if event["event_type"] == "Fixation":
                simplified_event["centroid_coordinates_label"] = event["centroid_coordinates_label"]
            elif event["event_type"] == "Saccade":
                simplified_event["direction_label"] = event["direction_label"]
                simplified_event["amplitude_label"] = event["amplitude_label"]
                # Optional to add speed level (symbol label) to avoid introducing specific values
                if "peak_velocity_label" in event:
                    simplified_event["peak_velocity_label"] = event["peak_velocity_label"]
            elif event["event_type"] == "SmoothPursuit":
                simplified_event["main_direction_label"] = event["main_direction_label"]
                simplified_event["travelled_distance_label"] = event["travelled_distance_label"]
                # Optional to add average speed level (symbol label) to avoid introducing specific values
                if "average_velocity_label" in event:
                    simplified_event["average_velocity_label"] = event["average_velocity_label"]
            
            simplified_events.append(simplified_event)
        
        return json.dumps(simplified_events, indent=2)
    
    def _generate_gaze_description(self, gaze_events: List[Dict]) -> str:
        """
        Generate gaze description based on gaze events sequence
        
        Args:
            gaze_events: Symbolized gaze events list, each item contains event_type, duration_label, direction information etc.
            
        Returns:
            str: Gaze description
        """
        # Create simplified events JSON string
        simplified_events_json = self._create_simplified_events_json(gaze_events)
        
        # Build prompt
        gaze_prompt = GAZE_DESCRIPTION_PROMPT.format(
            gaze_events=simplified_events_json  
        )
        
        # Generate gaze description
        gaze_description = self.gaze_llm.generate(gaze_prompt)
        
        return gaze_description, simplified_events_json

    def _prefetch_gaze_descriptions(self, items: List[Dict]) -> None:
        """Prefetch gaze descriptions in batch to improve throughput"""
        if self.batch_size <= 1 or not items:
            return

        pending: List[Tuple[str, str]] = []
        for item in items:
            motion_id = item['metadata']['motion_id']
            if motion_id in self.gaze_descriptions_cache:
                continue
            simplified_events_json = self._create_simplified_events_json(item['gaze_events'])
            gaze_prompt = GAZE_DESCRIPTION_PROMPT.format(gaze_events=simplified_events_json)
            pending.append((motion_id, gaze_prompt))

        if not pending:
            return

        supports_batch = hasattr(self.gaze_llm, "generate_batch")
        for i in range(0, len(pending), self.batch_size):
            batch = pending[i:i + self.batch_size]
            prompts = [entry[1] for entry in batch]

            if supports_batch:
                descriptions = self.gaze_llm.generate_batch(prompts)
                if not isinstance(descriptions, list) or len(descriptions) != len(batch):
                    logger.warning("Batch generation returned abnormal, downgrade to single generation")
                    descriptions = [self.gaze_llm.generate(p) for p in prompts]
            else:
                descriptions = [self.gaze_llm.generate(p) for p in prompts]

            for (motion_id, _), description in zip(batch, descriptions):
                self.gaze_descriptions_cache[motion_id] = description

    def _get_memory_context(self) -> str:
        """
        Get history memory context
        
        Returns:
            str: History memory context
        """
        # If history memory is not used or memory buffer is empty, return empty string
        if self.memory_use_count == 0 or not self.memory_buffer:
            return ""
        
        # Get recent n memories
        recent_memories = list(self.memory_buffer)[-self.memory_use_count:]
        
        memory_context = ""
        for i, memory in enumerate(recent_memories):
            timestamp_info = f"[Time: {memory['start_time']:.2f}-{memory['end_time']:.2f}] "
            memory_context += f"{i+1}. {timestamp_info}{memory['integrated_description']}\n"
        
        return memory_context
    
    def _evaluate_gaze_description(self, gaze_events: List[Dict], gaze_description: str, last_evaluation_result: str = None) -> Tuple[float, str, str]:
        """
        Evaluate gaze description
        # Note: Consider using gemini's function calling to evaluate gaze description, so the return value can be more accurate
        Args:
            gaze_events: Symbolized gaze events sequence
            gaze_description: Generated gaze description
            
        Returns:
            Tuple[float, str, str]: (score, evaluation feedback, suggestion)
        """
        # Convert gaze events to JSON string
        import json
        gaze_events_json = json.dumps(gaze_events, indent=2)
        
        # Build evaluation prompt
        evaluation_prompt = GAZE_EVALUATION_PROMPT.format(
            gaze_events=gaze_events_json,
            gaze_description=gaze_description,
            last_evaluation_result=last_evaluation_result
        )
        
        # Generate evaluation
        evaluation_result = self.gaze_evaluation_llm.generate(evaluation_prompt)
        
        # Parse evaluation result
        try:
            # Extract score
            continuity_score_match = re.search(r'Continuity Score:\s*(\d+(\.\d+)?)', evaluation_result)
            continuity_score = float(continuity_score_match.group(1)) if continuity_score_match else 0.0
            
            # Extract feedback - fix regex to ensure only extract content between Feedback and Suggestion
            feedback_match = re.search(r'Feedback:\s*(.*?)(?=\s*Suggestion:|$)', evaluation_result, re.DOTALL)
            feedback = feedback_match.group(1).strip() if feedback_match else ""
            
            # Extract suggestion
            suggestion_match = re.search(r'Suggestion:\s*(.*?)(?=$)', evaluation_result, re.DOTALL)
            suggestion = suggestion_match.group(1).strip() if suggestion_match else ""
            
            return continuity_score, feedback, suggestion
        except Exception as e:
            logger.error(f"Error parsing evaluation result: {str(e)}")
            return 0.0, evaluation_result, "Unable to parse suggestion"
    
    def _evaluate_integrated_description(
        self, 
        gaze_description: str, 
        body_posture: str, 
        focus_attention: Optional[str], 
        integrated_description: str,
        start_time: float,
        end_time: float,
        last_evaluation_result: str = None
    ) -> Tuple[float, Dict[str, float], str, str]:
        """
        Evaluate integrated description
        
        Args:
            gaze_description: Gaze description
            body_posture: Body posture description
            focus_attention: Focus attention description (optional, can be None)
            integrated_description: Generated integrated description
            start_time: Start time
            end_time: End time
            
        Returns:
            Tuple[float, Dict[str, float], str, str]: (overall score, dimension scores, evaluation feedback, suggestion)
        """
        # Build evaluation prompt
        memory_context_for_eval = ""
        if self.memory_use_count > 0 and self.memory_buffer:
            memory_context_for_eval = "Previous actions:\n" + self._get_memory_context()
        
        # Handle focus_attention is None case
        focus_attention_text = focus_attention if focus_attention is not None else "N/A"
        
        evaluation_prompt = INTEGRATED_EVALUATION_PROMPT.format(
            start_time=start_time,
            end_time=end_time,
            gaze_description=gaze_description,
            body_posture=body_posture,
            focus_attention=focus_attention_text,
            integrated_description=integrated_description,
            memory_context_for_eval=memory_context_for_eval,
            last_evaluation_result=last_evaluation_result
        )
        
        # Generate evaluation
        evaluation_result = self.integrated_evaluation_llm.generate(evaluation_prompt)
        
        # Parse evaluation result
        try:
            # Extract dimension scores
            match_score_match = re.search(r'Match Score:\s*(\d+(\.\d+)?)', evaluation_result)
            match_score = float(match_score_match.group(1)) if match_score_match else 0.0

            coherence_score_match = re.search(r'Temporal Coherence Score:\s*(\d+(\.\d+)?)', evaluation_result)
            coherence_score = float(coherence_score_match.group(1)) if coherence_score_match else 0.0
            
            completeness_score_match = re.search(r'Completeness Score:\s*(\d+(\.\d+)?)', evaluation_result)
            completeness_score = float(completeness_score_match.group(1)) if completeness_score_match else 0.0
            
            # Extract overall score, previously given by llm directly
            overall_score = round((match_score + coherence_score + completeness_score) / 3, 2) 
            
            # Extract feedback
            feedback_match = re.search(r'Feedback:\s*(.*?)(?=\s*Suggestion:|$)', evaluation_result, re.DOTALL)
            feedback = feedback_match.group(1).strip() if feedback_match else ""
            
            # Extract suggestion
            suggestion_match = re.search(r'Suggestion:\s*(.*?)(?=$)', evaluation_result, re.DOTALL)
            suggestion = suggestion_match.group(1).strip() if suggestion_match else ""
            
            scores = {
                'match': match_score,   
                'coherence': coherence_score,
                'completeness': completeness_score,
                'overall': overall_score
            }
            
            return overall_score, scores, feedback, suggestion
        except Exception as e:
            logger.error(f"Error parsing evaluation result: {str(e)}")
            return 0.0, {'match': 0, 'coherence': 0, 'completeness': 0, 'overall': 0}, evaluation_result, "Unable to parse suggestion"
    
    def _refine_gaze_description(
        self, 
        simplified_events_json: str, 
        original_description: str,
        evaluation_feedback: str,
        suggestion: str
    ) -> str:
        """
        Refine gaze description based on evaluation result
        
        Args:
            gaze_events: Symbolized gaze events sequence
            original_description: Original description
            evaluation_feedback: Evaluation feedback
            suggestion: Improvement suggestion
            
        Returns:
            str: Refined description
        """
        # Build refinement prompt
        refinement_prompt = GAZE_REFINEMENT_PROMPT.format(
            gaze_events=simplified_events_json,
            original_description=original_description,
            evaluation_feedback=f"{evaluation_feedback}\n\n{suggestion}",
        )
        
        # Generate refined description using dedicated refinement LLM
        refined_description = self.gaze_refinement_llm.generate(refinement_prompt)
        
        return refined_description
    
    def _refine_integrated_description(
        self,
        gaze_description: str,
        body_posture: str,
        focus_attention: Optional[str],
        memory_context: str,
        original_description: str,
        evaluation_feedback: str,
        suggestion: str,
        scores: Dict[str, float]
    ) -> str:
        """
        Refine integrated description based on evaluation result
        
        Args:
            gaze_description: Gaze description
            body_posture: Body posture description
            focus_attention: Focus attention description (optional, can be None)
            original_description: Original description
            evaluation_feedback: Evaluation feedback
            suggestion: Improvement suggestion
            scores: Dimension scores
            
        Returns:
            str: Refined description
        """
        # Build refinement prompt
        refinement_prompt = INTEGRATED_REFINEMENT_PROMPT.format(
            gaze_description=gaze_description,
            body_posture=body_posture,
            focus_attention=focus_attention,
            memory_context=memory_context,
            original_description=original_description,
            evaluation_feedback=f"{evaluation_feedback}\n\n{suggestion}",
        )
        
        # Generate refined description using dedicated refinement LLM
        refined_description = self.integrated_refinement_llm.generate(refinement_prompt)
        
        return refined_description
    
    def process_item(self, item: Dict) -> Dict:
        """
        Process single data item, including evaluation and refinement functionality
        
        Args:
            item: Symbolized gaze data item, containing metadata(motion_id, start_time, end_time, body_posture etc.)
                  and gaze_events(symbolized gaze events sequence) fields
            
        Returns:
            Dict: Processing result
        """
        try:
            # Extract data - support new symbolized gaze format
            motion_id = item['metadata']['motion_id']
            start_time = item['metadata']['start_time']
            end_time = item['metadata']['end_time']
            body_posture = item['metadata']['body_posture']
            focus_attention = item['metadata'].get('focus_attention', None)
            gaze_events = item['gaze_events']
            
            # 1. Generate and evaluate gaze description
            gaze_description = None
            gaze_evaluations = []
            retry_count = 0
            gaze_last_evaluation_result = ""
            # 1.1 Initial generation of gaze description
            # Check if the gaze description is already in the cache
            if motion_id in self.gaze_descriptions_cache:
                gaze_description = self.gaze_descriptions_cache[motion_id]
                simplified_events_json = self._create_simplified_events_json(gaze_events)
            else:
                gaze_description, simplified_events_json = self._generate_gaze_description(gaze_events)
                self.gaze_descriptions_cache[motion_id] = gaze_description
            
            # 1.2 Loop for evaluation and possible refinement
            while True:
                # Evaluate gaze description
                continuity_score, feedback, suggestion = self._evaluate_gaze_description(
                    gaze_events, gaze_description, gaze_last_evaluation_result
                )
                
                # Record evaluation result
                gaze_evaluation = {
                    'description': gaze_description,
                    'continuity_score': continuity_score,
                    'feedback': feedback,
                    'suggestion': suggestion,
                    'retry_count': retry_count
                }
                gaze_evaluations.append(gaze_evaluation)
                gaze_last_evaluation_result = (
                    f"Continuity Score: {continuity_score}\n"
                    f"Feedback: {feedback}\n"
                    f"Suggestion: {suggestion}\n"
                    f"Retry Count: {retry_count}"
                )   
                # 1.3 Check if refinement is needed
                if continuity_score < self.gaze_evaluation_threshold and retry_count < self.max_retry:
                    logger.info(f"Gaze description for motion_id {motion_id} scored {continuity_score}, below threshold {self.gaze_evaluation_threshold}. Retrying...")
                    gaze_description = self._refine_gaze_description(
                        simplified_events_json, gaze_description, feedback, suggestion
                    )
                    retry_count += 1
                    # If this is the last retry, stop evaluation, and use the refined result
                    if retry_count >= self.max_retry:
                        break
                else:
                    # If the score is达标或已达最大重试次数，结束循环
                    break
            
            # Cache the final gaze description
            self.gaze_descriptions_cache[motion_id] = gaze_description
            
            # 2. Get history memory context
            memory_context = self._get_memory_context()
            
            # 3. Generate and evaluate integrated description
            integrated_description = None
            integrated_evaluations = []
            retry_count = 0
            integrated_last_evaluation_result = ""
            # 3.1 Initial generation of integrated description
            # Select appropriate prompt template based on memory_use_count
            # Handle focus_attention is None case
            focus_attention_text = focus_attention if focus_attention is not None else "N/A"
            
            if self.memory_use_count == 0 or not memory_context:
                # Use prompt template without memory context
                integrated_prompt = INTEGRATED_DESCRIPTION_PROMPT_NO_MEMORY.format(
                    start_time=start_time,
                    end_time=end_time,
                    gaze_description=gaze_description,
                    body_posture=body_posture,
                    focus_attention=focus_attention_text
                )
            else:
                # Use prompt template with memory context
                integrated_prompt = INTEGRATED_DESCRIPTION_PROMPT_WITH_MEMORY.format(
                    start_time=start_time,
                    end_time=end_time,
                    memory_context=memory_context,
                    gaze_description=gaze_description,
                    body_posture=body_posture,
                    focus_attention=focus_attention_text
                )
            
            integrated_description = self.integrated_llm.generate(integrated_prompt)
            
            # 3.2 Loop for evaluation and possible refinement
            while True:
                # Evaluate integrated description
                overall_score, scores, feedback, suggestion = self._evaluate_integrated_description(
                    gaze_description, body_posture, focus_attention, integrated_description,
                    start_time, end_time, integrated_last_evaluation_result
                )
                
                # Record evaluation result
                integrated_evaluation = {
                    'description': integrated_description,
                    'overall_score': overall_score,
                    'dimension_scores': scores,
                    'feedback': feedback,
                    'suggestion': suggestion,
                    'retry_count': retry_count
                }
                integrated_evaluations.append(integrated_evaluation)
                # Convert integrated_evaluation content to text format and pass to last_evaluation_result
                integrated_last_evaluation_result = (
                    f"Dimension Scores: {scores}\n"
                    f"Feedback: {feedback}\n"
                    f"Suggestion: {suggestion}\n"
                    f"Retry Count: {retry_count}"
                )
                # 3.3 Check if refinement is needed
                if overall_score < self.integrated_evaluation_threshold and retry_count < self.max_retry:
                    logger.info(f"Integrated description for motion_id {motion_id} scored {overall_score}, below threshold {self.integrated_evaluation_threshold}. Retrying...")
                    integrated_description = self._refine_integrated_description(
                        gaze_description, body_posture, focus_attention,
                        memory_context, integrated_description, feedback, suggestion, scores
                    )
                    retry_count += 1
                    # If this is the last retry, stop evaluation, and use the refined result
                    if retry_count >= self.max_retry:
                        break
                else:
                    break
            
            # 4. Build result
            result = {
                'motion_id': motion_id,
                'start_time': start_time,
                'end_time': end_time,
                'body_posture': body_posture,
                'focus_attention': focus_attention,
                'gaze_description': gaze_description,
                'integrated_description': integrated_description,
                'gaze_evaluations': gaze_evaluations, 
                'integrated_evaluations': integrated_evaluations,
                'timestamp': datetime.now().isoformat()
            }
            
            # 5. Add result to memory buffer (whether history memory is used or not)
            memory_item = {
                'motion_id': motion_id,
                'start_time': start_time,
                'end_time': end_time,
                'body_posture': body_posture,
                'focus_attention': focus_attention,
                'gaze_description': gaze_description,
                'integrated_description': integrated_description,
                'timestamp': datetime.now().isoformat()
            }
            self.memory_buffer.append(memory_item)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing item with motion_id {item.get('metadata', {}).get('motion_id', 'unknown')}: {str(e)}")
            return {
                'motion_id': item.get('metadata', {}).get('motion_id', 'unknown'),
                'error': str(e)
            }
    
    def process_and_save_item(self, item: Dict, output_path: str) -> Dict:
        """
        Process single data item and save result
        
        Args:
            item: Symbolized gaze data item, containing metadata and gaze_events fields
            output_path: Output file path
            
        Returns:
            Dict: Processing result
        """
        if self.batch_size > 1 and item['metadata']['motion_id'] not in self.gaze_descriptions_cache:
            self._prefetch_gaze_descriptions([item])

        # Process data item
        result = self.process_item(item)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Read existing results (if exist)
        existing_results = []
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Unable to parse existing result file {output_path}, will create new file")
        
        # Update or add new result
        result_updated = False
        for i, existing_result in enumerate(existing_results):
            if existing_result.get('motion_id') == result.get('motion_id'):
                existing_results[i] = result
                result_updated = True
                break
        
        if not result_updated:
            existing_results.append(result)
        
        # Save updated result
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=2)
            
        # Save as CSV format
        csv_path = output_path.replace('.json', '.csv')
        simplified_results = []
        for r in existing_results:
            # Get the last evaluation score
            gaze_score = 0.0
            if r.get('gaze_evaluations'):
                gaze_score = r['gaze_evaluations'][-1].get('continuity_score', 0.0)
                
            integrated_score = 0.0
            if r.get('integrated_evaluations'):
                integrated_score = r['integrated_evaluations'][-1].get('overall_score', 0.0)
            
            simplified_result = {
                'motion_id': r.get('motion_id'),
                'start_time': r.get('start_time'),
                'end_time': r.get('end_time'),
                'body_posture': r.get('body_posture'),
                'focus_attention': r.get('focus_attention'),
                'gaze_description': r.get('gaze_description'),
                'integrated_description': r.get('integrated_description'),
                'gaze_score': gaze_score,
                'integrated_score': integrated_score,
                'timestamp': r.get('timestamp')
            }
            simplified_results.append(simplified_result)
            
        df = pd.DataFrame(simplified_results)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Processed and saved motion_id: {item['metadata']['motion_id']}")
        return result

    def process_all_items(self, max_items: Optional[int] = None, output_path: str = None) -> List[Dict]:
        """
        Process all data items, support checkpoint resume
        
        Args:
            max_items: Maximum number of data items to process
            output_path: Output file path, if provided, save after each data item is processed
            
        Returns:
            List[Dict]: Processing result list
        """
        # Get all data items from data processor
        all_items = self.data_processor.get_items(max_items)
        
        # If output path is provided, check if there are already processed data
        processed_motion_ids = set()
        existing_results = []
        
        if output_path and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                    # Record all processed motion_ids
                    for result in existing_results:
                        processed_motion_ids.add(result.get('motion_id'))
                    logger.info(f"Found number of processed data items: {len(processed_motion_ids)}")
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Unable to parse existing result file {output_path}: {str(e)}, will reprocess all data")
        
        # Filter out data items that are not processed
        items_to_process = [item for item in all_items if item['metadata']['motion_id'] not in processed_motion_ids]
        logger.info(f"Total data items: {len(all_items)}, processed: {len(processed_motion_ids)}, to process: {len(items_to_process)}")

        self._prefetch_gaze_descriptions(items_to_process)
        
        # Process data items that are not processed
        results = list(existing_results) 
        for idx, item in enumerate(items_to_process, start=1):
            if output_path:
                result = self.process_and_save_item(item, output_path)
            else:
                result = self.process_item(item)
            
            if not output_path:
                results.append(result)
            
            logger.info(f"Processed motion_id: {item['metadata']['motion_id']} ({idx}/{len(items_to_process)})")
            processed_motion_ids.add(item['metadata']['motion_id'])
        
        return results
    
    def process_items(self, items: List[Dict], output_path: str = None) -> List[Dict]:
        """
        Process specified data items list, support checkpoint resume
        
        Args:
            items: Data items list to process
            output_path: Output file path, if provided, save after each data item is processed
            
        Returns:
            List[Dict]: Processing result list
        """
        # If output path is provided, check if there are already processed data
        processed_motion_ids = set()
        existing_results = []
        
        if output_path and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                    # Record all processed motion_ids
                    for result in existing_results:
                        processed_motion_ids.add(result.get('motion_id'))
                    logger.info(f"Found number of processed data items: {len(processed_motion_ids)}")
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Unable to parse existing result file {output_path}: {str(e)}, will reprocess all data")
        
        # Filter out data items that are not processed
        items_to_process = [item for item in items if item['metadata']['motion_id'] not in processed_motion_ids]
        logger.info(f"Total data items: {len(items)}, processed: {len(processed_motion_ids)}, to process: {len(items_to_process)}")

        self._prefetch_gaze_descriptions(items_to_process)
        
        # Process data items that are not processed
        results = list(existing_results) 
        for i, item in enumerate(items_to_process):
            if output_path:
                result = self.process_and_save_item(item, output_path)
            else:
                result = self.process_item(item)
            
            # If process_and_save_item is not added to file, add to result list
            if not output_path:
                results.append(result)
            
            logger.info(f"Processed motion_id: {item['metadata']['motion_id']} ({i+1}/{len(items_to_process)})")
            processed_motion_ids.add(item['metadata']['motion_id'])
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str) -> None:
        """
        Save processing result
        
        Args:
            results: Processing result list
            output_path: Output file path
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Save as JSON format
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        # Save as CSV format
        simplified_results = []
        for result in results:
            # Get the last evaluation score
            gaze_score = 0.0
            if result.get('gaze_evaluations'):
                gaze_score = result['gaze_evaluations'][-1].get('continuity_score', 0.0)
                
            integrated_score = 0.0
            if result.get('integrated_evaluations'):
                integrated_score = result['integrated_evaluations'][-1].get('overall_score', 0.0)
            
            simplified_result = {
                'motion_id': result.get('motion_id'),
                'start_time': result.get('start_time'),
                'end_time': result.get('end_time'),
                'body_posture': result.get('body_posture'),
                'focus_attention': result.get('focus_attention'),
                'gaze_description': result.get('gaze_description'),
                'integrated_description': result.get('integrated_description'),
                'gaze_score': gaze_score,
                'integrated_score': integrated_score,
                'timestamp': result.get('timestamp')
            }
            simplified_results.append(simplified_result)
            
        csv_path = output_path.replace('.json', '.csv')
        df = pd.DataFrame(simplified_results)
        df.to_csv(csv_path, index=False)
            
        logger.info(f"Results saved to: {output_path} and {csv_path}")

    def test_evaluation_score_distribution(self, num_samples=5):
        """
        Test evaluation score distribution, to check if the evaluation model can give diverse scores
        
        Args:
            num_samples: Number of test samples
            
        Returns:
            Dict: Score distribution
        """
        logger.info("Start testing evaluation score distribution...")
        
        # 1. Test gaze description evaluation
        gaze_scores = []
        test_gaze_events = [
            {"event_type": "Fixation", "duration_label": "Short", "centroid_coordinates_label": "RightDown"},
            {"event_type": "Saccade", "duration_label": "Brief", "direction_label": "Up-Right", "amplitude_label": "Medium"},
            {"event_type": "Fixation", "duration_label": "Long", "centroid_coordinates_label": "RightUp"}
        ]
        
        # Test several descriptions of different quality
        test_descriptions = [
            # Excellent description
            "The human's gaze gradually shifts upward and to the right, smoothly tracking an object or person moving in that direction with focused attention.",
            # Average description
            "The human looks up and to the right, following something moving.",
            # Poor description
            "The human is looking somewhere with their eyes.",
            # Incorrect description
            "The human looks down and to the left, focusing on a stationary object.",
            # Very poor description
            "The human."
        ]
        
        for desc in test_descriptions:
            score, feedback, suggestion = self._evaluate_gaze_description(test_gaze_events, desc)
            gaze_scores.append({
                'description': desc,
                'score': score,
                'feedback': feedback,
                'suggestion': suggestion
            })
            logger.info(f"Gaze description: '{desc[:30]}...' score: {score}")
        
        # 2. Test integrated description evaluation
        integrated_scores = []
        test_gaze_description = "The human's gaze shifts from left to right, following an object in motion."
        test_body_posture = "Standing upright with arms at sides."
        test_focus_attention = "Looking at a computer screen."
        
        test_integrated_descriptions = [
            # Excellent description
            "The human stands with perfect posture, their arms comfortably at their sides while their head moves smoothly from left to right as they track an object moving across the computer screen, maintaining a steady gaze throughout the motion.",
            # Average description
            "The human stands and looks at the computer screen, moving their head from left to right.",
            # Poor description
            "The human looks at the screen while standing.",
            # Incorrect description
            "The human sits down and looks up at the ceiling.",
            # Very poor description
            "The human."
        ]
        
        for desc in test_integrated_descriptions:
            overall_score, scores, feedback, suggestion = self._evaluate_integrated_description(
                test_gaze_description, test_body_posture, test_focus_attention,
                desc, 0.0, 2.5
            )
            integrated_scores.append({
                'description': desc,
                'overall_score': overall_score,
                'dimension_scores': scores,
                'feedback': feedback,
                'suggestion': suggestion
            })
            logger.info(f"Integrated description: '{desc[:30]}...' score: {overall_score}")
        
        result = {
            'gaze_scores': gaze_scores,
            'integrated_scores': integrated_scores,
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate score statistics
        gaze_score_values = [item['score'] for item in gaze_scores]
        integrated_score_values = [item['overall_score'] for item in integrated_scores]
        
        logger.info(f"Gaze description score statistics: minimum={min(gaze_score_values)}, maximum={max(gaze_score_values)}, average={sum(gaze_score_values)/len(gaze_score_values)}")
        logger.info(f"Integrated description score statistics: minimum={min(integrated_score_values)}, maximum={max(integrated_score_values)}, average={sum(integrated_score_values)/len(integrated_score_values)}")
        
        return result
