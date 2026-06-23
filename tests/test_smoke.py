import unittest
import importlib

class SmokeTest(unittest.TestCase):
    def test_import_agents(self):
        modules = [
            'agents.perception_agent',
            'agents.environment_agent',
            'agents.decision_agent',
            'agents.reasoning_agent',
            'agents.safety_agent',
        ]
        for m in modules:
            try:
                importlib.import_module(m)
            except Exception as e:
                self.fail(f"Failed to import {m}: {e}")

if __name__ == '__main__':
    unittest.main()
