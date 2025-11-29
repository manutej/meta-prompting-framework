"""
Meta-Prompting Framework - Signature System

Provides typed input/output specifications for prompts.
Based on DSPy's signature system with categorical enhancements.
"""

from typing import Any, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Field:
    """
    A typed field in a signature.

    Defines the type, description, and validation for prompt I/O.
    """

    type: type
    description: str = ""
    required: bool = True
    default: Any = None
    name: str = ""  # Set during Signature initialization

    def validate(self, value: Any) -> bool:
        """Validate value against type."""
        try:
            if self.type == int:
                return isinstance(value, int)
            elif self.type == float:
                return isinstance(value, (int, float))
            elif self.type == str:
                return isinstance(value, str)
            elif self.type == List[str]:
                return isinstance(value, list) and all(isinstance(x, str) for x in value)
            elif self.type == list:
                return isinstance(value, list)
            else:
                return isinstance(value, self.type)
        except:
            return False


class InputField(Field):
    """Input field to a prompt."""
    pass


class OutputField(Field):
    """Output field from a prompt."""
    pass


class Signature:
    """
    A signature defines the input/output structure of a prompt.

    Signatures provide type safety and structured I/O for prompt modules.
    Similar to DSPy's Signature, but integrated with categorical abstractions.

    Example:
        class QASignature(Signature):
            '''Answer questions with reasoning.'''
            question = InputField(str, "The question to answer")
            reasoning = OutputField(str, "Step-by-step reasoning")
            answer = OutputField(str, "Final answer")

        # Usage
        signature = QASignature()
        signature.validate_input({"question": "What is 2+2?"})
        prompt = signature.format_prompt({"question": "What is 2+2?"})
        outputs = signature.parse_output(llm_response)
    """

    def __init__(self):
        self._input_fields: Dict[str, InputField] = {}
        self._output_fields: Dict[str, OutputField] = {}
        self._instruction = self.__class__.__doc__ or ""

        # Extract fields from class attributes
        for name in dir(self.__class__):
            if name.startswith('_'):
                continue
            value = getattr(self.__class__, name)
            if isinstance(value, InputField):
                field = InputField(
                    type=value.type,
                    description=value.description,
                    required=value.required,
                    default=value.default,
                    name=name
                )
                self._input_fields[name] = field
            elif isinstance(value, OutputField):
                field = OutputField(
                    type=value.type,
                    description=value.description,
                    required=value.required,
                    default=value.default,
                    name=name
                )
                self._output_fields[name] = field

    @property
    def instruction(self) -> str:
        """Get the signature instruction (from docstring)."""
        return self._instruction

    @property
    def input_fields(self) -> Dict[str, InputField]:
        """Get all input fields."""
        return self._input_fields

    @property
    def output_fields(self) -> Dict[str, OutputField]:
        """Get all output fields."""
        return self._output_fields

    def validate_input(self, inputs: Dict[str, Any]) -> bool:
        """Validate input dictionary against input fields."""
        for name, field in self._input_fields.items():
            if field.required and name not in inputs:
                raise ValueError(f"Required input field '{name}' missing")
            if name in inputs and not field.validate(inputs[name]):
                raise TypeError(
                    f"Input field '{name}' has wrong type. "
                    f"Expected {field.type}, got {type(inputs[name])}"
                )
        return True

    def validate_output(self, outputs: Dict[str, Any]) -> bool:
        """Validate output dictionary against output fields."""
        for name, field in self._output_fields.items():
            if field.required and name not in outputs:
                raise ValueError(f"Required output field '{name}' missing")
            if name in outputs and not field.validate(outputs[name]):
                raise TypeError(
                    f"Output field '{name}' has wrong type. "
                    f"Expected {field.type}, got {type(outputs[name])}"
                )
        return True

    def format_prompt(self, inputs: Dict[str, Any]) -> str:
        """
        Generate prompt string from inputs.

        Creates a structured prompt with instruction and formatted inputs.
        """
        self.validate_input(inputs)

        prompt_parts = [self.instruction.strip(), ""]

        # Add input fields
        for name, field in self._input_fields.items():
            value = inputs.get(name, field.default)
            label = field.description or name.replace('_', ' ').title()
            prompt_parts.append(f"{label}: {value}")

        # Add output field labels
        prompt_parts.append("")
        for name, field in self._output_fields.items():
            label = field.description or name.replace('_', ' ').title()
            prompt_parts.append(f"{label}:")

        return "\n".join(prompt_parts)

    def parse_output(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into output fields.

        Simple heuristic parsing that looks for field labels in the response.
        Can be overridden for more complex parsing needs.
        """
        outputs = {}
        current_field = None
        current_value = []

        for line in response.split("\n"):
            line_stripped = line.strip()

            # Check if line starts a new field
            field_matched = False
            for name, field in self._output_fields.items():
                label = field.description or name.replace('_', ' ').title()

                # Try exact match with colon
                if line_stripped.startswith(f"{label}:"):
                    # Save previous field
                    if current_field:
                        outputs[current_field] = "\n".join(current_value).strip()
                    # Start new field
                    current_field = name
                    remainder = line_stripped[len(label)+1:].strip()
                    current_value = [remainder] if remainder else []
                    field_matched = True
                    break

            # Continue current field if no new field matched
            if not field_matched and current_field:
                current_value.append(line)

        # Save last field
        if current_field:
            outputs[current_field] = "\n".join(current_value).strip()

        self.validate_output(outputs)
        return outputs


# Common Signatures

class ChainOfThoughtSignature(Signature):
    """Answer questions with step-by-step reasoning."""

    question = InputField(str, "Question")
    reasoning = OutputField(str, "Step-by-step reasoning")
    answer = OutputField(str, "Final answer")


class RAGSignature(Signature):
    """Answer questions using retrieved context."""

    question = InputField(str, "Question")
    context = InputField(List[str], "Retrieved passages")
    answer = OutputField(str, "Answer based on context")
    citations = OutputField(List[str], "Source citations", required=False)


class CodeGenerationSignature(Signature):
    """Generate code with explanation."""

    task = InputField(str, "Programming task")
    language = InputField(str, "Programming language")
    code = OutputField(str, "Generated code")
    explanation = OutputField(str, "Code explanation")


class MathSignature(Signature):
    """Solve mathematical problems."""

    problem = InputField(str, "Math problem")
    solution_steps = OutputField(str, "Step-by-step solution")
    final_answer = OutputField(str, "Final answer")


class DebugSignature(Signature):
    """Debug code and provide fixes."""

    code = InputField(str, "Code with bugs")
    error_message = InputField(str, "Error message", required=False)
    diagnosis = OutputField(str, "Bug diagnosis")
    fixed_code = OutputField(str, "Corrected code")
    explanation = OutputField(str, "Explanation of fix")
