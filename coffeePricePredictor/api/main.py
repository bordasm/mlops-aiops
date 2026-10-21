"""Start the coffee quality REST API on localhost."""

import uvicorn

from coffee_api import config
from coffee_api.app import app


def main() -> None:
    """Run the API server with uvicorn."""
    uvicorn.run(app, host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    main()
