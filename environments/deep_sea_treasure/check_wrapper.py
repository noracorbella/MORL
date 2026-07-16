"""
Check that the Deep Sea Treasure wrapper satisfies the MOEnv contract.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from environments.deep_sea_treasure.wrapper import DeepSeaTreasureEnv
from morl.core.validate_env import check_moenv_contract


def main():
    env = DeepSeaTreasureEnv()
    passed = check_moenv_contract(env)
    env.close()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
