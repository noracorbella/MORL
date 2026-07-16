"""
Check that the Resource Gathering wrapper satisfies the MOEnv contract.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from environments.resource_gathering.wrapper import ResourceGatheringEnv
from morl.core.validate_env import check_moenv_contract


def main():
    env = ResourceGatheringEnv()
    passed = check_moenv_contract(env)
    env.close()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
