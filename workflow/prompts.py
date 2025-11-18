# Prompt Templates

###### System prompts ######
GAZE_SYSTEM_INSTRUCTION = """You are a Human Gaze Analysis Expert. Your goal is to convert symbolic gaze event sequences into a single, continuous, human-centric sentence that highlights the key gaze patterns for downstream text-to-gaze training.

# Input semantics (from symbolic gaze reference)
- Event types: Fixation, Saccade, SmoothPursuit
- Labels: duration_label ∈ {Brief, Short, Moderate, Long}; direction/centroid labels (e.g., Left, Up-Right, Center); amplitude_label ∈ {Small, Medium, Large}; (optional) speed labels: peak_velocity_label for Saccade, average_velocity_label for SmoothPursuit
- Coordinate convention: yaw > 0 is right; pitch > 0 is up (do not mention numbers)

# Controlled vocabulary
- Use the exact tokens when referring to event types: "fixation", "saccade", "smooth pursuit" (avoid synonyms like glance, gaze shift, tracking)
- Use qualitative words implied by labels only: duration → brief/short/moderate/long; amplitude → small/medium/large; speed → slow/fast/very fast (pursuit may use steady)

# Style & constraints
- Summarize only the most salient patterns; do not enumerate every minor micro-event
- Maintain temporal continuity across gaze shifts
- Output one sentence under 40 words
- Start with "The human..."; no numbers, no counts, no coordinates, no code fences, no explanations
"""

GAZE_REFINEMENT_SYSTEM_INSTRUCTION = """You are a Gaze Description Refinement Specialist. Improve the gaze description according to evaluation feedback while fully preserving the global rules.

# Global rules
- Keep controlled vocabulary for event types: "fixation", "saccade", "smooth pursuit"
- Rely on labels (duration/direction/amplitude/speed) for qualitative wording; do not invent numbers or counts
- Ensure temporal continuity and natural flow; summarize key patterns only
- Output one sentence under 40 words, starting with "The human..."; no code fences, no extra commentary
"""

INTEGRATED_SYSTEM_INSTRUCTION = """You are an expert in human motion understanding and description. Generate integrated, temporally coherent motion narratives suitable for text-to-motion generation and intent analysis.

# Guidelines
- Naturally integrate gaze behavior, body posture, and focus of attention
- Prioritize physical motions and posture details; include only gaze details relevant to motion/intent
- Ensure temporal continuity throughout
- Output under 50 words, starting with "The human..."; avoid numbers
"""

INTEGRATED_REFINEMENT_SYSTEM_INSTRUCTION = """You are a Motion Description Specialist. Refine integrated behavioral descriptions per evaluation feedback while preserving all constraints.

# Guidelines
- Improve MATCH (integration), TEMPORAL COHERENCE, and COMPLETENESS
- Keep majority of detail on motion/posture; retain only gaze details directly relevant to motion or intent
- Output under 50 words, starting with "The human..."; avoid numbers and extra commentary
"""

GAZE_EVALUATION_SYSTEM_INSTRUCTION = """You are a Gaze Description Evaluator. Your job is to assess how well a one-sentence gaze description matches a symbolic gaze event sequence.

# Scoring target
- Output exactly three sections in order: (1) Continuity Score: [0-5]; (2) Feedback: ...; (3) Suggestion: ...
- Continuity = temporal coherence + correctness of major transitions + focus on key patterns (avoid trivial enumerations)
- If a previous evaluation exists, compare the current description against the previously flagged issues. If the issues are addressed, increase the score relative to the previous one; if not, keep the same or reduce. Avoid keeping identical scores when clear improvements are made.

# Use of numeric fields
- The gaze event JSON may include numeric timestamps, durations, angles, amplitudes, and velocities. Use these ONLY to check label consistency and ordering (e.g., duration_label vs duration, amplitude_label vs amplitude, speed labels vs velocities, direction_label vs displacement vector, event order via timestamps). Do not output any numbers in your result.

# Feedback & Suggestion
- Feedback: 2–3 sentences that (a) identify mismatches or missing key transitions, (b) mention over-enumeration if present, and (c) state whether it improves over the previous evaluation (if provided).
- Suggestion: Provide a single revised one-sentence description starting with "The human..." that fixes the issues, using controlled vocabulary (fixation / saccade / smooth pursuit) and qualitative labels only.

# Constraints
- Respect the coordinate convention: yaw > 0 is right; pitch > 0 is up
- Encourage controlled vocabulary usage in the candidate description: fixation / saccade / smooth pursuit
- Do not require or reward numbers, coordinates, or counts; evaluate qualitative label alignment (duration/amplitude/speed/direction)
- Do not add extra sections beyond the required format; no code fences
"""

INTEGRATED_EVALUATION_SYSTEM_INSTRUCTION = """You are an Integrated Motion Description Evaluator. Assess an integrated motion description against gaze, posture, attention, and (if provided) previous motion descriptions.

# Scoring target
- Output exactly: Match Score: [0-5]; Temporal Coherence Score: [0-5]; Completeness Score: [0-5]; Feedback: ...; Suggestion: ...
- MATCH = how well gaze/posture/attention are naturally combined; TEMPORAL COHERENCE = logical timeline and consistency with previous actions; COMPLETENESS = important elements from inputs are captured
- If a previous evaluation exists, raise scores when the new description addresses prior feedback; keep the same or reduce if issues persist. Mention this comparison explicitly in Feedback.

# Feedback & Suggestion
- Feedback: 2–3 sentences referencing MATCH, TEMPORAL COHERENCE, and COMPLETENESS; explicitly state whether there is improvement vs the last evaluation if provided.
- Suggestion: Provide a single revised one-sentence integrated description starting with "The human...", concise and motion-focused.

# Constraints
- Prefer motion/posture details; only gaze details directly relevant to motion/intent should be emphasized
- Avoid numbers and technical jargon; no extra sections; no code fences
"""

###### User prompts ######
GAZE_DESCRIPTION_PROMPT = """Analyze the following gaze event sequence and generate a natural one-sentence description.

# One shot example
Input gaze events:
[
  {{"event_type": "Fixation", "duration_label": "Long", "centroid_coordinates_label": "LeftDown"}},
  {{"event_type": "Saccade",  "duration_label": "Brief", "direction_label": "Up-Right", "amplitude_label": "Large", "peak_velocity_label": "Fast"}},
  {{"event_type": "Fixation", "duration_label": "Short", "centroid_coordinates_label": "RightUp"}}
]
Output:
The human holds a long fixation at lower-left, then makes a fast, large saccade up-right to a brief fixation there.

# Gaze event sequence to analyze (sorted by timestamp):
{gaze_events}

# Output Requirements
- Focus on key patterns; do not list trivial events
- Use controlled vocabulary: fixation / saccade / smooth pursuit; avoid synonyms
- Use qualitative words implied by labels (duration, amplitude, speed) only; do not invent numbers or counts
- Maintain temporal coherence; one sentence under 40 words; start with "The human..."

# Output:"""

INTEGRATED_DESCRIPTION_PROMPT_NO_MEMORY = """Integrate the following gaze behavior, body posture, and focus attention into a detailed motion description.

Time frame: {start_time:.2f} to {end_time:.2f} seconds

# Input components
- Gaze behavior: {gaze_description}
- Body posture: {body_posture}
- Focus attention: {focus_attention}

# Output Requirements
- Prioritize motion and posture details; include only gaze details relevant to the motion/intent
- Avoid numbers; maintain temporal continuity
- One sentence under 50 words, starting with "The human..."

# Output:"""

INTEGRATED_DESCRIPTION_PROMPT_WITH_MEMORY = """Integrate the following gaze behavior, body posture, and focus attention into a detailed motion description that maintains continuity with previous motion descriptions.

Time frame: {start_time:.2f} to {end_time:.2f} seconds

# Previous motion descriptions (for context)
{memory_context}

# Current input components
- Gaze behavior: {gaze_description}
- Body posture: {body_posture}
- Focus attention: {focus_attention}

# Output Requirements
- Prioritize motion and posture details; include only gaze details relevant to the motion/intent
- Avoid numbers; maintain temporal continuity with previous actions
- One sentence under 50 words, starting with "The human..."

# Output:"""


###### Evaluation prompts ######
GAZE_EVALUATION_PROMPT = """Evaluate the quality of the given gaze description against the gaze event sequence and provide feedback and suggestions.

# One-shot examples
## Example 1 (no previous evaluation)
Input:
Gaze event sequence (sorted by timestamp)
[
  {{"event_type":"Fixation","duration_label":"Long","centroid_coordinates_label":"LeftDown","start_time":0.00,"end_time":0.80}},
  {{"event_type":"Saccade","duration_label":"Brief","direction_label":"Up-Right","amplitude_label":"Large","peak_velocity":320,"start_time":0.80,"end_time":0.84}},
  {{"event_type":"Fixation","duration_label":"Short","centroid_coordinates_label":"RightUp","start_time":0.84,"end_time":1.10}}
]
Generated gaze description
The human looks left, then quickly moves up-right and glances briefly.
Last evaluation result (if exists)
None
Output:
Continuity Score: 3
Feedback: Captures the main transitions and ordering, but uses synonyms instead of the controlled vocabulary and lacks qualitative labels (long fixation, large fast saccade, brief fixation). Direction is correct; transition is present but could be clearer.
Suggestion: The human holds a long fixation at lower-left, then makes a fast, large saccade up-right to a brief fixation.

## Example 2 (with previous evaluation)
Input:
Gaze event sequence (sorted by timestamp)
...
Generated gaze description
...
Last evaluation result (if exists)
Continuity Score: 3
Feedback: Replace synonyms with fixation/saccade and add qualitative labels for duration and speed.
Suggestion: The human holds a long fixation at lower-left, then makes a fast, large saccade up-right to a brief fixation.
New candidate description:
The human holds a long fixation at lower-left, then makes a fast, large saccade up-right to a brief fixation.
Output:
Continuity Score: 4
Feedback: The description adopts the controlled vocabulary and qualitative labels as suggested, improving clarity and continuity over the previous attempt. Transitions and directions align with events and timestamps.
Suggestion: The human holds a long fixation at lower-left, then makes a fast, large saccade up-right to a brief fixation.

# Expected output format
Continuity Score: [0-5]
Feedback: [...]
Suggestion: [...]

# Gaze event sequence (sorted by timestamp)
{gaze_events}

# Generated gaze description
{gaze_description}

# Last evaluation result (if exists)
{last_evaluation_result}

# Requirements
- Coordinate convention: yaw > 0 is right, pitch > 0 is up
- Penalize enumerations of trivial events; reward continuity and focus on key patterns
- Reward correct use of controlled vocabulary (fixation/saccade/smooth pursuit) and qualitative labels; do not require numbers
- If a previous evaluation exists, adjust the score relative to the previous one based on whether flagged issues are fixed; avoid identical scores when clear improvements exist

# Your evaluation:"""

INTEGRATED_EVALUATION_PROMPT = """Evaluate the quality of the given integrated motion description and provide feedback and suggestions.

# One-shot examples
Components:
- Gaze behavior: The human holds a long fixation at lower-left, then makes a fast, large saccade up-right to a brief fixation.
- Body posture: Standing upright with arms at sides.
- Focus attention: Looking at a computer screen.
Candidate integrated description:
The human stands and looks at the screen, moving the head from left to right.
Output:
Match Score: 3
Temporal Coherence Score: 3
Completeness Score: 3
Feedback: Captures posture and a general head movement but omits the key gaze transitions and their relation to the task. Temporal links are weak and do not reference salient changes.
Suggestion: The human stands upright, focusing on the screen, then shifts gaze with a fast, large saccade up-right before a brief fixation, while keeping arms relaxed at the sides.

## Example 2 (with previous evaluation)
Last evaluation result:
Match Score: 3
Temporal Coherence Score: 3
Completeness Score: 3
Feedback: Include the key gaze transition and tighten temporal flow.
Suggestion: The human stands upright, focusing on the screen, then shifts gaze with a fast, large saccade up-right before a brief fixation, while keeping arms relaxed at the sides.
New candidate integrated description:
The human stands upright, focusing on the screen, shifts gaze with a fast, large saccade up-right to a brief fixation, maintaining relaxed arms.
Output:
Match Score: 4
Temporal Coherence Score: 4
Completeness Score: 4
Feedback: Addresses prior feedback by adding the salient gaze transition and improving temporal flow; posture and attention remain consistent.
Suggestion: The human stands upright, focusing on the screen, shifts gaze with a fast, large saccade up-right to a brief fixation, maintaining relaxed arms.

# Expected output format
Match Score: [0-5]
Temporal Coherence Score: [0-5]
Completeness Score: [0-5]
Feedback: [...]
Suggestion: [...]

Time frame: {start_time:.2f} to {end_time:.2f} seconds

# Components
- Gaze behavior: {gaze_description}
- Body posture: {body_posture}
- Focus attention: {focus_attention}
- Previous motion descriptions: {memory_context_for_eval}

# Generated integrated motion description
{integrated_description}

# Last evaluation result (if exists)
{last_evaluation_result}

# Requirements
- Ensure scoring reflects integration quality, temporal coherence, and completeness
- Prefer concise, motion-focused descriptions with only relevant gaze details
- If a previous evaluation exists, explicitly compare against it in Feedback and adjust all relevant scores accordingly; raise scores when specific prior issues are addressed, otherwise keep the same or lower. Avoid identical scores when clear improvements exist.

# Your evaluation:"""


###### Refinement prompts ######
GAZE_REFINEMENT_PROMPT = """Refine the original gaze description based on the evaluation feedback and suggestions.

# Gaze event sequence (input for generating the original gaze description)
{gaze_events}

# Original gaze description
{original_description}

# Evaluation feedback and suggestions
{evaluation_feedback}

# Requirements
- Keep controlled vocabulary (fixation/saccade/smooth pursuit); avoid synonyms and numbers
- Address all specific feedback while maintaining a single, coherent sentence under 40 words
- Summarize only the key patterns; start with "The human..."

# Provide only the revised description:"""

INTEGRATED_REFINEMENT_PROMPT = """Refine the original integrated motion description based on the evaluation feedback and suggestions.

# Components used to generate the original description
- Gaze behavior: {gaze_description}
- Body posture: {body_posture}
- Focus attention: {focus_attention}
- Previous motion descriptions (if any): {memory_context}

# Original integrated motion description
{original_description}

# Evaluation feedback and suggestions
{evaluation_feedback}

# Requirements
- Improve integration, temporal continuity, and completeness; keep majority of detail on motion/posture
- Avoid numbers; one sentence under 50 words; start with "The human..."
- Provide only the revised description (no preface, no code fences)

# Provide only the revised description:"""
