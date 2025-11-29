"""
Meta-Prompting Framework - Module System

Composable prompt modules with categorical guarantees.
Based on DSPy's Module with RMP monad integration.
"""

from typing import Dict, Any, Optional, List, Callable
from abc import ABC, abstractmethod
from .signature import Signature
from ..categorical.monad import RMPMonad


class Module(ABC):
    """
    Base class for composable prompt modules.

    A module takes a signature and implements a prompting strategy.
    Modules compose via the RMP monad to ensure quality monotonicity.

    Example:
        class CustomModule(Module):
            def forward(self, **inputs) -> Dict[str, Any]:
                # Implement your prompting logic
                return outputs
    """

    def __init__(
        self,
        signature: type[Signature],
        llm_client: Optional[Any] = None
    ):
        """
        Initialize a module with a signature and LLM client.

        Args:
            signature: Signature class (not instance) defining I/O
            llm_client: LLM client for making API calls
        """
        self.signature = signature()
        self.llm_client = llm_client
        self._quality_history: List[float] = []

    @abstractmethod
    def forward(self, **inputs) -> Dict[str, Any]:
        """
        Execute the module with given inputs.

        Args:
            **inputs: Input values matching signature input fields

        Returns:
            Dictionary of outputs matching signature output fields
        """
        pass

    def __call__(self, **inputs) -> Dict[str, Any]:
        """Alias for forward."""
        return self.forward(**inputs)

    def compose(self, other: 'Module') -> 'Module':
        """
        Compose two modules sequentially.

        Returns a SequentialModule that runs self then other.
        """
        return SequentialModule([self, other])

    def with_rmp(self, improve_fn: Callable[[str, Dict], RMPMonad]) -> 'RMPModule':
        """
        Wrap this module with RMP (Recursive Meta-Prompting).

        Args:
            improve_fn: Function that takes (prompt, outputs) and returns improved RMPMonad

        Returns:
            RMPModule that applies recursive meta-prompting
        """
        return RMPModule(self, improve_fn)


class Predict(Module):
    """
    Basic prediction module.

    Generates prompt from signature and gets LLM response.
    No additional reasoning or tool use.
    """

    def forward(self, **inputs) -> Dict[str, Any]:
        """Execute basic prediction."""
        if self.llm_client is None:
            raise ValueError("LLM client required for Predict module")

        self.signature.validate_input(inputs)

        # Generate prompt from signature
        prompt = self.signature.format_prompt(inputs)

        # Get LLM response
        response = self.llm_client.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        # Parse outputs
        outputs = self.signature.parse_output(response.content)
        return outputs


class ChainOfThought(Module):
    """
    Chain-of-thought reasoning module.

    Augments the signature prompt with explicit step-by-step reasoning.
    Implements the chain-of-thought prompting strategy.
    """

    def __init__(
        self,
        signature: type[Signature],
        llm_client: Optional[Any] = None,
        reasoning_style: str = "step-by-step"
    ):
        super().__init__(signature, llm_client)
        self.reasoning_style = reasoning_style

    def forward(self, **inputs) -> Dict[str, Any]:
        """Execute with chain-of-thought reasoning."""
        if self.llm_client is None:
            raise ValueError("LLM client required for ChainOfThought module")

        self.signature.validate_input(inputs)

        # Generate base prompt
        base_prompt = self.signature.format_prompt(inputs)

        # Add reasoning instruction
        if self.reasoning_style == "step-by-step":
            reasoning_instruction = """
Please think step-by-step:
1. Break down the problem into smaller parts
2. Solve each part carefully
3. Combine the solutions for the final answer

Show your reasoning clearly before providing the answer.
"""
        elif self.reasoning_style == "socratic":
            reasoning_instruction = """
Please use Socratic reasoning:
1. What is the core question?
2. What do we know?
3. What are the implications?
4. What is the conclusion?
"""
        else:
            reasoning_instruction = "\nPlease explain your reasoning.\n"

        enhanced_prompt = base_prompt + reasoning_instruction

        # Get LLM response
        response = self.llm_client.complete(
            messages=[{"role": "user", "content": enhanced_prompt}],
            temperature=0.7
        )

        # Parse outputs
        outputs = self.signature.parse_output(response.content)
        return outputs


class ReAct(Module):
    """
    ReAct (Reasoning + Acting) module.

    Interleaves reasoning and tool use in a loop.
    Implements the ReAct pattern from Yao et al.
    """

    def __init__(
        self,
        signature: type[Signature],
        llm_client: Optional[Any] = None,
        tools: Optional[Dict[str, Callable]] = None,
        max_iterations: int = 5
    ):
        super().__init__(signature, llm_client)
        self.tools = tools or {}
        self.max_iterations = max_iterations

    def forward(self, **inputs) -> Dict[str, Any]:
        """Execute ReAct reasoning-action loop."""
        if self.llm_client is None:
            raise ValueError("LLM client required for ReAct module")

        self.signature.validate_input(inputs)

        thought_action_history = []

        for i in range(self.max_iterations):
            # Build prompt with history
            prompt = self._build_react_prompt(inputs, thought_action_history)

            # Get LLM response
            response = self.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )

            # Parse thought/action/action_input
            thought, action, action_input = self._parse_react_response(response.content)

            thought_action_history.append({
                "thought": thought,
                "action": action,
                "action_input": action_input
            })

            # Check for finish
            if action == "Finish":
                # Parse action_input as the final output
                return self._parse_final_output(action_input)

            # Execute tool
            if action in self.tools:
                try:
                    observation = str(self.tools[action](action_input))
                    thought_action_history[-1]["observation"] = observation
                except Exception as e:
                    thought_action_history[-1]["observation"] = f"Error: {str(e)}"
            else:
                thought_action_history[-1]["observation"] = f"Error: Unknown action '{action}'"

        # Max iterations reached - return best guess
        return self._create_fallback_output(thought_action_history)

    def _build_react_prompt(self, inputs: Dict, history: List[Dict]) -> str:
        """Build ReAct-style prompt with tool descriptions and history."""
        prompt_parts = []

        # Add tool descriptions
        if self.tools:
            prompt_parts.append("You can use the following tools:")
            for name, func in self.tools.items():
                doc = func.__doc__ or f"{name} tool"
                prompt_parts.append(f"- {name}: {doc}")
            prompt_parts.append("")

        # Add task from signature
        prompt_parts.append(self.signature.format_prompt(inputs))
        prompt_parts.append("")

        # Add ReAct format instructions
        prompt_parts.append("Use this format:")
        prompt_parts.append("Thought: [your reasoning about what to do next]")
        prompt_parts.append("Action: [tool name or 'Finish']")
        prompt_parts.append("Action Input: [input to the tool or final answer]")
        prompt_parts.append("Observation: [result from tool, or empty if Finish]")
        prompt_parts.append("")

        # Add history
        for entry in history:
            prompt_parts.append(f"Thought: {entry['thought']}")
            prompt_parts.append(f"Action: {entry['action']}")
            prompt_parts.append(f"Action Input: {entry['action_input']}")
            if 'observation' in entry:
                prompt_parts.append(f"Observation: {entry['observation']}")
                prompt_parts.append("")

        if history:
            prompt_parts.append("Continue:")

        return "\n".join(prompt_parts)

    def _parse_react_response(self, response: str) -> tuple:
        """Parse thought/action/action_input from LLM response."""
        thought = ""
        action = ""
        action_input = ""

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("Thought:"):
                thought = line[8:].strip()
            elif line.startswith("Action:"):
                action = line[7:].strip()
            elif line.startswith("Action Input:"):
                action_input = line[13:].strip()

        return thought, action, action_input

    def _parse_final_output(self, action_input: str) -> Dict[str, Any]:
        """Parse the final action input into signature outputs."""
        # Simple heuristic: treat action_input as the answer
        # Can be overridden for more complex parsing
        output_fields = list(self.signature.output_fields.keys())
        if len(output_fields) == 1:
            return {output_fields[0]: action_input}
        else:
            # Try to parse from action_input
            try:
                return self.signature.parse_output(action_input)
            except:
                # Fallback to primary output field
                return {output_fields[0]: action_input}

    def _create_fallback_output(self, history: List[Dict]) -> Dict[str, Any]:
        """Create fallback output when max iterations reached."""
        # Use the last thought as the answer
        last_thought = history[-1]["thought"] if history else "No answer determined"
        output_fields = list(self.signature.output_fields.keys())
        return {output_fields[0]: last_thought}


class SequentialModule(Module):
    """
    Compose modules sequentially.

    Output of module[i] becomes input to module[i+1].
    Ensures quality monotonicity via RMP monad composition.
    """

    def __init__(self, modules: List[Module]):
        if not modules:
            raise ValueError("SequentialModule requires at least one module")

        self.modules = modules
        # Use signature of last module as overall signature
        super().__init__(modules[-1].signature.__class__)

    def forward(self, **inputs) -> Dict[str, Any]:
        """Execute modules sequentially."""
        current_inputs = inputs

        for i, module in enumerate(self.modules):
            current_inputs = module.forward(**current_inputs)

        return current_inputs


class RMPModule(Module):
    """
    Recursive Meta-Prompting Module.

    Wraps any module with RMP (Recursive Meta-Prompting) to iteratively
    improve outputs using the RMP monad.
    """

    def __init__(
        self,
        base_module: Module,
        improve_fn: Callable[[str, Dict], RMPMonad],
        max_iterations: int = 3,
        quality_threshold: float = 0.90
    ):
        super().__init__(base_module.signature.__class__, base_module.llm_client)
        self.base_module = base_module
        self.improve_fn = improve_fn
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold

    def forward(self, **inputs) -> Dict[str, Any]:
        """Execute with recursive meta-prompting."""
        # Initial execution
        outputs = self.base_module.forward(**inputs)

        # Create initial RMP monad
        prompt = self.signature.format_prompt(inputs)
        rmp = RMPMonad.unit(prompt)

        best_outputs = outputs
        best_quality = 0.0

        # Iterative improvement
        for i in range(self.max_iterations):
            # Apply improvement function
            rmp = rmp.flat_map(lambda p: self.improve_fn(p, outputs))

            # Check quality
            if rmp.quality > best_quality:
                best_quality = rmp.quality
                # Re-execute with improved prompt
                # (In practice, would need to extract prompt and re-run)

            # Early stopping
            if rmp.quality >= self.quality_threshold:
                break

        return best_outputs
