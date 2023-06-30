"""Run evaluator wrapper for string evaluators."""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchainplus_sdk import EvaluationResult, RunEvaluator
from langchainplus_sdk.schemas import Example, Run

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (AsyncCallbackManagerForChainRun,
                                         CallbackManagerForChainRun)
from langchain.chains.base import Chain
from langchain.evaluation.schema import StringEvaluator
from langchain.load.serializable import Serializable
from langchain.schema import RUN_KEY, get_buffer_string, messages_from_dict
from langchain.tools.base import Tool


class StringRunMapper(Serializable):
    """Extract items to evaluate from the run object."""

    @property
    def output_keys(self) -> List[str]:
        """The keys to extract from the run."""
        raise ["prediction", "input"]
    
    @abstractmethod
    def map(self, run: Run) -> Dict[str, Any]:
        """Maps the Run to a dictionary."""

    def __call__(self, run: Run) -> Dict[str, Any]:
        """Maps the Run to a dictionary."""
        if not run.outputs:
            raise ValueError(f"Run {run.id} has no outputs to evaluate.")
        return self.map(run)

class LLMStringRunMapper(StringRunMapper):
    """Extract items to evaluate from the run object."""

    
    def serialize_chat_messages(self, messages: List[Dict]) -> str:
        """Extract the input messages from the run."""
        chat_messages = messages_from_dict(messages)
        return get_buffer_string(chat_messages)
    
    def serialize_inputs(self, inputs: Dict) -> str:
        if "prompts" in inputs: # Should we even accept this?
            input_ = "\n\n".join(inputs["prompts"])
        elif "prompt" in inputs:
            input_ = inputs["prompt"]
        elif "messages" in inputs:
            input_ = self.serialize_chat_messages(inputs["messages"])
        else:
            raise ValueError("LLM Run must have either messages or prompts as inputs.")
        return input_
    
    def serialize_outputs(self, outputs: Dict) -> str:
        if not outputs.get("generations"):
            raise ValueError("LLM Run must have generations as outputs.")
        generations: List[Dict] = outputs["generations"]
        if 'messages' in generations[0]:
            # Chat model
            output_ = self.serialize_chat_messages(generations[0]["messages"])
        else:
            output_ = "\n\n".join([generation["text"] for generation in generations])
        return output_

    
    def map(self, run: Run) -> Dict[str, Any]:
        """Maps the Run to a dictionary."""
        if run.run_type != "llm":
            raise ValueError("LLM RunMapper only supports LLM runs.")
        elif not run.outputs:
            if run.error:
                raise ValueError(f"Cannot evaluate errored LLM run {run.id}: {run.error}")
            else:
                raise ValueError(f"Run {run.id} outputs not found. Please make sure this")
        else:
            inputs = self.serialize_inputs(run.inputs)
            output_ = self.serialize_outputs(run.outputs)
            return {"input": inputs, "prediction": output_}

class ChainStringRunMapper(StringRunMapper):
    """Extract items to evaluate from the run object from a chain."""

    input_key: str
    """The key from the model Run's inputs to use as the input to the evaluation."""
    prediction_key: str
    """The key from the model Run's outputs to use as the prediction to the evaluation."""

    @classmethod
    def from_chain(cls, model: Chain, input_key: Optional[str] = None, prediction_key: Optional[str] = None) -> ChainStringRunMapper:
        """Create a RunMapper from a chain."""
        error_messages = []
        if input_key is None:
            if len(model.input_keys) > 1:
                error_messages.append(f"Chain {model.lc_namespace} has multiple input keys, so you must specify one.")
            else:
                input_key = model.input_keys[0]
        elif input_key not in model.input_keys:
            error_messages.append(f"Chain {model.lc_namespace} does not have specified input key {input_key}.")
        if prediction_key is None:
            if len(model.output_keys) > 1:
                error_messages.append(f"Chain {model.lc_namespace} has produces multiple outputs, so a prediction_key is required.")
            else:
                prediction_key = model.output_keys[0]
        elif prediction_key not in model.output_keys:
            error_messages.append(f"Chain {model.lc_namespace} does not have specified prediction_key {prediction_key}.")
        if error_messages:
            raise ValueError("\n".join(error_messages))
        return cls(input_key=input_key, prediction_key=prediction_key)

    def map(self, run: Run) -> Dict[str, Any]:
        """Maps the Run to a dictionary."""
        if run.run_type != "chain":
            raise ValueError("Chain RunMapper only supports Chain runs.")
        if self.input_key not in run.inputs:
            raise ValueError(f"Run {run.id} does not have input key {self.input_key}.")
        elif self.prediction_key not in run.outputs:
            raise ValueError(f"Run {run.id} does not have prediction key {self.prediction_key}.")
        else:
            return {"input": run.inputs[self.input_key], "prediction": run.outputs[self.prediction_key]}


class ToolStringRunMapper(StringRunMapper):
    """Map an input to the tool."""

    def map(self, run: Run) -> Dict[str, Any]:
        return {"input": run.inputs["input"], "prediction": run.outputs["output"]}

    


class StringExampleMapper(Serializable):
    """Map an example, or row in the dataset, to the inputs of an evaluation."""

    reference_key: str
    
    @property
    def output_keys(self) -> List[str]:
        """The keys to extract from the run."""
        raise ["reference"]
    
    def map(self, example: Example) -> Dict[str, Any]:
        """Maps the Example, or dataset row to a dictionary."""
        try:
            return example[self.reference_key]
        except KeyError:
            raise ValueError(
                f"Could not evaluate run. Example {example.id} does not"
                f" have reference key {self.reference_key}.")

    def __call__(self, example: Example) -> Dict[str, Any]:
        """Maps the Run and Example to a dictionary."""
        if not example.outputs:
            raise ValueError(f"Example {example.id} has no outputs to use as areference label.")
        return self.map(example)
    

class StringRunEvaluatorChain(Chain, RunEvaluator):
    """Evaluate Run and optional examples."""

    run_mapper: StringRunMapper
    """Maps the Run to a dictionary for the eval chain."""
    example_mapper: Optional[StringExampleMapper] = None
    """Maps the Example (dataset row) to a dictionary for the eval chain."""
    name: str
    """The name of the evaluation metric."""
    string_evaluator: StringEvaluator
    """The evaluation chain."""

    @property
    def input_keys(self) -> List[str]:
        return ["run", "example"]

    @property
    def output_keys(self) -> List[str]:
        return ["feedback"]
    
    def _get_evaluation_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        run: Run = inputs["run"]
        example: Optional[Example] = inputs.get("example")
        evaluate_strings_inputs = self.run_mapper(run)
        if self.example_mapper:
            if not example:
                raise ValueError(
                    f"Evaluator {self.name} requires an reference example from the dataset,"
                    f" but none was provided for run {run.id}."
                )
            evaluate_strings_inputs.update(self.example_mapper(example))
        return evaluate_strings_inputs

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Call the evaluation chain."""
        evaluate_strings_inputs = self._get_evaluation_inputs(inputs) 
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        chain_output = self.string_evaluator.evaluate_strings(
            **evaluate_strings_inputs, callbacks=callbacks, include_run_info=True
        )
        evaluation_result = EvaluationResult(**chain_output)
        evaluation_result.evaluator_info[RUN_KEY] = chain_output[RUN_KEY]
        return {"feedback": evaluation_result}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> Dict[str, Any]:
        """Call the evaluation chain."""
        evaluate_strings_inputs = self._get_evaluation_inputs(inputs)
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        chain_output = await self.string_evaluator.aevaluate_strings(
            **evaluate_strings_inputs,
            callbacks=callbacks,
            include_run_info=True,
        )
        run_info = chain_output[RUN_KEY]
        feedback = EvaluationResult(**chain_output)
        feedback.evaluator_info[RUN_KEY] = run_info
        return {"feedback": feedback}

    def evaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> EvaluationResult:
        """Evaluate an example."""
        return self({"run": run, "example": example})["feedback"]

    async def aevaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> EvaluationResult:
        """Evaluate an example."""
        result = await self.acall({"run": run, "example": example})
        return result["feedback"]

    @classmethod
    def from_model_and_evaluator(cls, model: Union[Chain, BaseLanguageModel, Tool], evaluator: StringEvaluator, prediction_key: Optional[str] = None, reference_key: Optional[str] = None) -> StringRunEvaluatorChain:
        """Create a StringRunEvaluatorChain from a model and evaluator."""
        if isinstance(model, BaseLanguageModel):
            run_mapper = LLMStringRunMapper()
        elif isinstance(model, Chain):
            run_mapper = ChainStringRunMapper.from_chain(model, prediction_key=prediction_key)
        elif isinstance(model, Tool):
            run_mapper = ToolStringRunMapper()
        else:
            raise NotImplementedError(
                f"{cls.__name__}.from_model_and_evaluator({type(model)}) not yet implemented."
                "Expected one of [BaseLanguageModel, Chain, Tool].")
        
        if reference_key is not None:
            example_mapper = StringExampleMapper(reference_key)
        else:
            example_mapper = None

        return cls(
            run_mapper=run_mapper,
            example_mapper=example_mapper,
            evaluator=evaluator,
        )
        
