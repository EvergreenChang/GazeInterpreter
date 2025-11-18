import logging
import sys
import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Dict

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_agents.gemini import GeminiLLM
from llm_agents.openai import OpenAILLM
from llm_agents.deepseek import DeepseekLLM
from llm_agents.llama import LlamaLLM
from workflow.data_processor import DataProcessor
from workflow.gazemotion_agents import GazeMotionAgent
from workflow.prompts import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationProfile:
    temperature: float
    max_tokens: int
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None

    def to_local_kwargs(self) -> Dict[str, float]:
        kwargs: Dict[str, float] = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.repetition_penalty is not None:
            kwargs["repetition_penalty"] = self.repetition_penalty
        return kwargs

    def to_api_config(self) -> Dict[str, float]:
        config: Dict[str, float] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            config["top_p"] = self.top_p
        return config

# Generation profiles for different agents
GENERATION_PROFILES: Dict[str, GenerationProfile] = {
    "gaze": GenerationProfile(temperature=0.7, max_tokens=100, top_p=0.9, repetition_penalty=1.05),
    "gaze_refine": GenerationProfile(temperature=0.6, max_tokens=100, top_p=0.85, repetition_penalty=1.05),
    "integrated": GenerationProfile(temperature=0.7, max_tokens=128, top_p=0.9, repetition_penalty=1.05),
    "integrated_refine": GenerationProfile(temperature=0.5, max_tokens=128, top_p=0.85, repetition_penalty=1.05),
    "evaluation": GenerationProfile(temperature=0.5, max_tokens=192, top_p=0.8, repetition_penalty=1.05),
}


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Gaze-Motion Agent Processing Program")
    parser.add_argument("--llm_provider", type=str, default="gemini",
                        choices=["gemini", "openai", "deepseek", "llama", "gemma_local"],
                        help="Select LLM provider: gemini, openai, deepseek, llama or gemma_local")
    parser.add_argument("--api_key", type=str, help="LLM API key (local model can be ignored)")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Input data path (can be a JSON file or a folder containing JSON files)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output result path (if input is a folder, this will be treated as output directory)")
    parser.add_argument("--max_items", type=int, default=None, help="Maximum number of data pairs to process")
    parser.add_argument("--memory_use_count", type=int, default=2, help="Number of historical memories to use")
    parser.add_argument("--memory_buffer_size", type=int, default=10, help="Memory buffer size")
    parser.add_argument("--gaze_eval_threshold", type=float, default=4.5, help="Gaze description evaluation threshold")
    parser.add_argument("--integrated_eval_threshold", type=float, default=4.5, help="Integrated description evaluation threshold")
    parser.add_argument("--max_retry", type=int, default=2, help="Maximum number of retries")
    parser.add_argument("--test_score_distribution", action="store_true", help="Test score distribution mode")
    parser.add_argument("--model", type=str, help="Specify model name")
    parser.add_argument("--model_path", type=str, help="Local model path or Hugging Face repository name")
    parser.add_argument("--torch_dtype", type=str, default=None, help="Dtype to load model with, example: bfloat16, float16, auto")
    parser.add_argument("--device_map", type=str, default="auto", help="transformers device_map parameter, default is auto")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation, e.g. flash_attention_2")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of gaze descriptions to generate in batch, only valid for local models")
    parser.add_argument("--filter_files", type=str, nargs='+', default=None,
                        help="List of JSON file names to filter (without .json extension), e.g. --filter_files 20230911_s0_angela_gomez_act3_8wxz7b 20230913_s0_stacey_lamb_act4_qp828r")
    
    args = parser.parse_args()
    
    # Configure parameters
    LLM_PROVIDER = args.llm_provider
    API_KEY = args.api_key or os.environ.get(f"{LLM_PROVIDER.upper()}_API_KEY", "")
    INPUT_DATA_PATH = args.input_path
    OUTPUT_PATH = args.output_path
    MAX_ITEMS = args.max_items
    MEMORY_USE_COUNT = args.memory_use_count
    MEMORY_BUFFER_SIZE = args.memory_buffer_size
    GAZE_EVALUATION_THRESHOLD = args.gaze_eval_threshold
    INTEGRATED_EVALUATION_THRESHOLD = args.integrated_eval_threshold
    MAX_RETRY = args.max_retry
    TEST_SCORE_DISTRIBUTION = args.test_score_distribution
    MODEL_PATH = args.model_path or os.environ.get("GEMMA_LOCAL_MODEL_PATH")
    TORCH_DTYPE = args.torch_dtype or os.environ.get("GEMMA_LOCAL_TORCH_DTYPE") or "bfloat16"
    DEVICE_MAP = args.device_map
    ATTN_IMPLEMENTATION = args.attn_implementation or os.environ.get("GEMMA_LOCAL_ATTN_IMPL")
    BATCH_SIZE = max(1, args.batch_size or int(os.environ.get("GEMMA_LOCAL_BATCH_SIZE", 1)))
    # FILTER_FILES = args.filter_files
    # FILTER_FILES = []
    FILTER_FILES = None
    # Set default model based on LLM provider
    if args.model:
        MODEL = args.model
    else:
        if LLM_PROVIDER == "gemini":
            MODEL = "gemini-2.5-flash"   # gemini-2.5-flash, gemini-2.0-flash,gemini-2.5-flash-preview-05-20
        elif LLM_PROVIDER == "openai":
            MODEL = "gpt-4o"
        elif LLM_PROVIDER == "deepseek":
            MODEL = "deepseek-chat"
        elif LLM_PROVIDER == "llama":
            MODEL = "llama-4-maverick-17b-128e-instruct"
        else:
            MODEL = args.model_path or MODEL_PATH or "google/gemma-3-4b-it"
    
    try:
        # Initialize LLM based on provider
        gaze_profile = GENERATION_PROFILES["gaze"]
        gaze_refine_profile = GENERATION_PROFILES["gaze_refine"]
        integrated_profile = GENERATION_PROFILES["integrated"]
        integrated_refine_profile = GENERATION_PROFILES["integrated_refine"]
        evaluation_profile = GENERATION_PROFILES["evaluation"]

        if LLM_PROVIDER == "gemini":
            # Initialize Gemini LLM
            gaze_llm = GeminiLLM(
                api_key=API_KEY,
                model="gemini-2.0-flash",
                reasoning_llm=False,
                system_instruction=GAZE_SYSTEM_INSTRUCTION,
                temperature=gaze_profile.temperature,
                thinking_budget=0,
                generation_config=gaze_profile.to_api_config(),
            )
            
            gaze_refinement_llm = GeminiLLM(
                api_key=API_KEY,
                model=MODEL,
                reasoning_llm=True,
                system_instruction=GAZE_REFINEMENT_SYSTEM_INSTRUCTION,
                temperature=gaze_refine_profile.temperature,
                thinking_budget=0,
                generation_config=gaze_refine_profile.to_api_config(),
            )
            
            integrated_llm = GeminiLLM(
                api_key=API_KEY,
                model="gemini-2.0-flash",
                reasoning_llm=False,
                system_instruction=INTEGRATED_SYSTEM_INSTRUCTION,
                temperature=integrated_profile.temperature,
                thinking_budget=0,
                generation_config=integrated_profile.to_api_config(),
            )
            
            integrated_refinement_llm = GeminiLLM(
                api_key=API_KEY,
                model=MODEL,
                reasoning_llm=False,
                system_instruction=INTEGRATED_REFINEMENT_SYSTEM_INSTRUCTION,
                temperature=integrated_refine_profile.temperature,
                thinking_budget=0,
                generation_config=integrated_refine_profile.to_api_config(),
            )
            
            gaze_evaluation_llm = GeminiLLM(
                api_key=API_KEY,
                model=MODEL,
                reasoning_llm=True,
                system_instruction=GAZE_EVALUATION_SYSTEM_INSTRUCTION,
                temperature=evaluation_profile.temperature,
                thinking_budget=0,
                generation_config=evaluation_profile.to_api_config(),
            )
            
            integrated_evaluation_llm = GeminiLLM(
                api_key=API_KEY,
                model=MODEL,
                reasoning_llm=True,
                system_instruction=INTEGRATED_EVALUATION_SYSTEM_INSTRUCTION,
                temperature=evaluation_profile.temperature,
                thinking_budget=0,
                generation_config=evaluation_profile.to_api_config(),
            )
        elif LLM_PROVIDER == "openai":
            # Initialize OpenAI LLM
            gaze_llm = OpenAILLM(
                api_key=API_KEY,
                model=MODEL,
                system_instruction=GAZE_SYSTEM_INSTRUCTION,
                generation_config=gaze_profile.to_api_config()
            )
            
            gaze_refinement_llm = OpenAILLM(
                api_key=API_KEY,
                model=MODEL,
                system_instruction=GAZE_REFINEMENT_SYSTEM_INSTRUCTION,
                generation_config=gaze_refine_profile.to_api_config()
            )
            
            integrated_llm = OpenAILLM(
                api_key=API_KEY,
                model=MODEL,
                system_instruction=INTEGRATED_SYSTEM_INSTRUCTION,
                generation_config=integrated_profile.to_api_config()
            )
            
            integrated_refinement_llm = OpenAILLM(
                api_key=API_KEY,
                model=MODEL,
                system_instruction=INTEGRATED_REFINEMENT_SYSTEM_INSTRUCTION,
                generation_config=integrated_refine_profile.to_api_config()
            )
            
            gaze_evaluation_llm = OpenAILLM(
                api_key=API_KEY,
                model=MODEL,
                system_instruction=GAZE_EVALUATION_SYSTEM_INSTRUCTION,
                generation_config=evaluation_profile.to_api_config()
            )
            
            integrated_evaluation_llm = OpenAILLM(
                api_key=API_KEY,
                model=MODEL,
                system_instruction=INTEGRATED_EVALUATION_SYSTEM_INSTRUCTION,
                generation_config=evaluation_profile.to_api_config()
            )
        elif LLM_PROVIDER == "deepseek":
            # Initialize Deepseek LLM
            gaze_llm = DeepseekLLM(
                api_key=API_KEY,
                model=MODEL,
                system_instruction=GAZE_SYSTEM_INSTRUCTION,
                generation_config=gaze_profile.to_api_config()
            )
            
            gaze_refinement_llm = DeepseekLLM(
                api_key=API_KEY,
                model=MODEL,
                system_instruction=GAZE_REFINEMENT_SYSTEM_INSTRUCTION,
                generation_config=gaze_refine_profile.to_api_config()
            )
            
            integrated_llm = DeepseekLLM(
                api_key=API_KEY,
                model=MODEL,
                system_instruction=INTEGRATED_SYSTEM_INSTRUCTION,
                generation_config=integrated_profile.to_api_config()
            )
            
            integrated_refinement_llm = DeepseekLLM(
                api_key=API_KEY,
                model=MODEL,
                system_instruction=INTEGRATED_REFINEMENT_SYSTEM_INSTRUCTION,
                generation_config=integrated_refine_profile.to_api_config()
            )
            
            gaze_evaluation_llm = DeepseekLLM(
                api_key=API_KEY,
                model=MODEL,
                system_instruction=GAZE_EVALUATION_SYSTEM_INSTRUCTION,
                generation_config=evaluation_profile.to_api_config()
            )

            integrated_evaluation_llm = DeepseekLLM(
                api_key=API_KEY,
                model=MODEL,
                system_instruction=INTEGRATED_EVALUATION_SYSTEM_INSTRUCTION,
                generation_config=evaluation_profile.to_api_config()
            )
        
        elif LLM_PROVIDER == "llama":  # llama - Bailei platform
            # Initialize Bailei platform Llama LLM
            gaze_llm = LlamaLLM(
                api_key=API_KEY,
                model=MODEL,
                system_instruction=GAZE_SYSTEM_INSTRUCTION,
                generation_config=gaze_profile.to_api_config()
            )
            
            gaze_refinement_llm = LlamaLLM(
                api_key=API_KEY,
                model=MODEL,
                system_instruction=GAZE_REFINEMENT_SYSTEM_INSTRUCTION,
                generation_config=gaze_refine_profile.to_api_config()
            )
            
            integrated_llm = LlamaLLM(
                api_key=API_KEY,
                model=MODEL,
                system_instruction=INTEGRATED_SYSTEM_INSTRUCTION,
                generation_config=integrated_profile.to_api_config()
            )
            
            integrated_refinement_llm = LlamaLLM(
                api_key=API_KEY,
                model=MODEL,
                system_instruction=INTEGRATED_REFINEMENT_SYSTEM_INSTRUCTION,
                generation_config=integrated_refine_profile.to_api_config()
            )
            
            gaze_evaluation_llm = LlamaLLM(
                api_key=API_KEY,
                model=MODEL,
                system_instruction=GAZE_EVALUATION_SYSTEM_INSTRUCTION,
                generation_config=evaluation_profile.to_api_config()
            )

            integrated_evaluation_llm = LlamaLLM(
                api_key=API_KEY,
                model=MODEL,
                system_instruction=INTEGRATED_EVALUATION_SYSTEM_INSTRUCTION,
                generation_config=evaluation_profile.to_api_config()
            )
        elif LLM_PROVIDER == "gemma_local":
            from llm_agents.gemma_local import GemmaLocalLLM

            model_path = MODEL_PATH or "google/gemma-3-4b-it"

            logger.info(
                "Using local Gemma model: %s (dtype=%s, device_map=%s, attn_impl=%s, batch_size=%s)",
                model_path,
                TORCH_DTYPE,
                DEVICE_MAP,
                ATTN_IMPLEMENTATION or "default",
                BATCH_SIZE,
            )

            shared_kwargs = {
                "model_path": model_path,
                "torch_dtype": TORCH_DTYPE,
                "device_map": DEVICE_MAP,
                "attn_implementation": ATTN_IMPLEMENTATION,
                "use_chat_template": True,
            }

            gaze_llm = GemmaLocalLLM(
                system_instruction=GAZE_SYSTEM_INSTRUCTION,
                **shared_kwargs,
                **gaze_profile.to_local_kwargs(),
            )

            gaze_refinement_llm = GemmaLocalLLM(
                system_instruction=GAZE_REFINEMENT_SYSTEM_INSTRUCTION,
                **shared_kwargs,
                **gaze_refine_profile.to_local_kwargs(),
            )

            integrated_llm = GemmaLocalLLM(
                system_instruction=INTEGRATED_SYSTEM_INSTRUCTION,
                **shared_kwargs,
                **integrated_profile.to_local_kwargs(),
            )

            integrated_refinement_llm = GemmaLocalLLM(
                system_instruction=INTEGRATED_REFINEMENT_SYSTEM_INSTRUCTION,
                **shared_kwargs,
                **integrated_refine_profile.to_local_kwargs(),
            )

            gaze_evaluation_llm = GemmaLocalLLM(
                system_instruction=GAZE_EVALUATION_SYSTEM_INSTRUCTION,
                **shared_kwargs,
                **evaluation_profile.to_local_kwargs(),
            )

            integrated_evaluation_llm = GemmaLocalLLM(
                system_instruction=INTEGRATED_EVALUATION_SYSTEM_INSTRUCTION,
                **shared_kwargs,
                **evaluation_profile.to_local_kwargs(),
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

        # Initialize data processor
        data_processor = DataProcessor(INPUT_DATA_PATH)

        # Initialize agent
        agent = GazeMotionAgent(
            gaze_llm=gaze_llm,
            gaze_refinement_llm=gaze_refinement_llm,
            integrated_llm=integrated_llm,
            integrated_refinement_llm=integrated_refinement_llm,
            gaze_evaluation_llm=gaze_evaluation_llm,
            integrated_evaluation_llm=integrated_evaluation_llm,
            data_processor=data_processor,
            memory_use_count=MEMORY_USE_COUNT,
            memory_buffer_size=MEMORY_BUFFER_SIZE,
            gaze_evaluation_threshold=GAZE_EVALUATION_THRESHOLD,
            integrated_evaluation_threshold=INTEGRATED_EVALUATION_THRESHOLD,
            max_retry=MAX_RETRY,
            batch_size=BATCH_SIZE,
        )
        
        # Process data
        if TEST_SCORE_DISTRIBUTION:
            # Test score distribution mode
            logger.info("Test score distribution mode")
            test_results = agent.test_evaluation_score_distribution()
            
            # Save test results
            test_output_path = OUTPUT_PATH.replace('.json', '_test_scores.json')
            os.makedirs(os.path.dirname(test_output_path) if os.path.dirname(test_output_path) else '.', exist_ok=True)
            with open(test_output_path, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Test score distribution results saved to: {test_output_path}")
        else:
            # Normal processing of data
            is_directory = os.path.isdir(INPUT_DATA_PATH)
            
            if is_directory:
                # If input is a directory, ensure output path is also a directory
                os.makedirs(OUTPUT_PATH, exist_ok=True)
                
                # Get all file data
                all_items = data_processor.get_items(max_items=MAX_ITEMS)
                
                # Apply file filtering
                if FILTER_FILES is not None:
                    filtered_items = {}
                    available_files = list(all_items.keys())
                    logger.info(f"Available files: {available_files}")
                    logger.info(f"Filter conditions: {FILTER_FILES}")
                    
                    for filename, items in all_items.items():
                        # Check if file name (without .json extension) is in filter list
                        base_filename = os.path.splitext(filename)[0]
                        
                        # Check if any name in filter list is contained in file name
                        should_include = any(filter_name in base_filename for filter_name in FILTER_FILES)
                        
                        if should_include:
                            filtered_items[filename] = items
                            logger.info(f"Include file: {filename} (match filter conditions)")
                        else:
                            logger.info(f"Skip file: {filename} (not match filter conditions)")
                    
                    if not filtered_items:
                        logger.warning(f"No matching file found based on filter conditions {FILTER_FILES}")
                        logger.info("Please check if the filter conditions are correct, or use the following available file names:")
                        for filename in available_files:
                            base_name = os.path.splitext(filename)[0]
                            logger.info(f"  - {base_name}")
                        return
                    
                    all_items = filtered_items
                    logger.info(f"After filtering, {len(all_items)} files will be processed")
                
                # Process each file
                for filename, items in all_items.items():
                    # Generate output file name
                    base_filename = os.path.splitext(filename)[0]
                    output_file = os.path.join(OUTPUT_PATH, f"{base_filename}_result.json")
                    
                    logger.info(f"Process file: {filename}, output to: {output_file}")
                    
                    # Process data for single file
                    results = agent.process_items(items, output_file)
                    logger.info(f"File {filename} processed! Results saved to: {output_file}")
            else:
                # Process data for single file
                results = agent.process_all_items(max_items=MAX_ITEMS, output_path=OUTPUT_PATH)
                logger.info(f"Processing completed! Results saved to: {OUTPUT_PATH}")
        
    except Exception as e:
        logger.error(f"Error executing program: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
