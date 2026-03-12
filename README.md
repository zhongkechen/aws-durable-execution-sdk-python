# Async Durable Execution SDK for Python

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

-----

Build AWS Lambda Durable Functions With Async Python

## ✨ Key Features

- **Automatic checkpointing** - Resume execution after Lambda pauses or restarts
- **Durable steps** - Run work with retry strategies and deterministic replay
- **Waits and callbacks** - Pause for time or external signals without blocking Lambda
- **Parallel and map operations** - Fan out work with configurable completion criteria
- **Child contexts** - Structure complex workflows into isolated subflows
- **Replay-safe logging** - Use `context.logger` for structured, de-duplicated logs
- **Local and cloud testing** - Validate workflows with the testing SDK

## 🚀 Quick Start

Install the execution SDK:

```console
pip install async-durable-execution
```

Create a durable Lambda handler:

```python
from async_durable_execution import (
    DurableContext,
    StepContext,
    durable_execution,
    durable_step,
)
from async_durable_execution.config import Duration


@durable_step
async def validate_order(step_ctx: StepContext, order_id: str) -> dict:
    step_ctx.logger.info("Validating order", extra={"order_id": order_id})
    return {"order_id": order_id, "valid": True}


@durable_execution
async def handler(event: dict, context: DurableContext) -> dict:
    order_id = event["order_id"]
    context.logger.info("Starting workflow", extra={"order_id": order_id})

    validation = await context.step(validate_order(order_id), name="validate_order")
    if not validation["valid"]:
        return {"status": "rejected", "order_id": order_id}

    # simulate approval (real world: use wait_for_callback)
    await context.wait(duration=Duration.from_seconds(5), name="await_confirmation")

    return {"status": "approved", "order_id": order_id}
```

## 📚 Documentation

- **[AWS Documentation](https://docs.aws.amazon.com/lambda/latest/dg/durable-functions.html)** - Official AWS Lambda durable functions guide

## 💬 Feedback & Support

- [Contributing guide](CONTRIBUTING.md)

## 📄 License

See the [LICENSE](LICENSE) file for our project's licensing.
