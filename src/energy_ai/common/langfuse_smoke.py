from dotenv import load_dotenv
load_dotenv()

from energy_ai.common.observability import get_langfuse_client


def main():
    lf = get_langfuse_client()
    if not lf:
        raise RuntimeError("Langfuse client not created. Check env vars.")

    # Create a trace using whatever API exists
    if hasattr(lf, "trace"):
        t = lf.trace(name="smoke-test-ford-energy", input={"ping": "pong"})
        if hasattr(t, "generation"):
            t.generation(name="gen", model="none", input={"x": 1}, output={"y": 2})
        if hasattr(lf, "flush"):
            lf.flush()
        print("sent via lf.trace")
        return

    raise RuntimeError("This Langfuse SDK does not support lf.trace(). Upgrade/downgrade needed.")


if __name__ == "__main__":
    main()