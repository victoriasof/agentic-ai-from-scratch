from mcp.server.fastmcp import FastMCP

# Create the server — this name identifies it to any client that connects
mcp = FastMCP("calculator-server")


@mcp.tool()
def calculator(a: float, b: float, operation: str) -> str:
    """Performs basic arithmetic: add, subtract, multiply, or divide two numbers.

    Args:
        a: The first number
        b: The second number
        operation: One of "add", "subtract", "multiply", "divide"
    """
    if operation == "add":
        return str(a + b)
    elif operation == "subtract":
        return str(a - b)
    elif operation == "multiply":
        return str(a * b)
    elif operation == "divide":
        return str(a / b)
    return "Unknown operation"


if __name__ == "__main__":
    mcp.run()