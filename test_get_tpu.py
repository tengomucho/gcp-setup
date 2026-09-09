import importlib.util
import json
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner


spec = importlib.util.spec_from_file_location(
    "get_tpu", Path(__file__).with_name("get-tpu.py")
)
get_tpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(get_tpu)

ZONE = "europe-west4-a"
NAME = f"amoran-tpu-dev-{ZONE}"


def node(name, state="READY", ip="192.0.2.1"):
    return {
        "name": f"projects/test-project/locations/{ZONE}/nodes/{name}",
        "state": state,
        "networkEndpoints": [{"accessConfig": {"externalIp": ip}}],
    }


class TpuNameMatchingTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.output = self.stack.enter_context(patch.object(get_tpu, "_gcloud_output"))

    def invoke_create(self, nodes):
        self.output.return_value = json.dumps(nodes)
        self.stack.enter_context(patch.object(get_tpu, "get_cache", return_value={}))
        self.stack.enter_context(
            patch.object(get_tpu, "get_project", return_value="test-project")
        )
        self.stack.enter_context(
            patch.object(
                get_tpu,
                "get_config",
                return_value=get_tpu.Config(tpu_name_prefix="amoran-tpu-dev-"),
            )
        )
        self.command = self.stack.enter_context(patch.object(get_tpu, "_run"))
        self.save = self.stack.enter_context(patch.object(get_tpu, "save_cache"))
        self.install = self.stack.enter_context(
            patch.object(get_tpu, "install_tpu_script")
        )
        result = CliRunner().invoke(
            get_tpu.app,
            [
                "create",
                "--accelerator-type",
                "v6e-4",
                "--software-version",
                "v2-alpha-tpuv6e",
                "--location",
                ZONE,
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        return result

    def test_create_ignores_unrelated_and_suffix_matching_names(self):
        self.invoke_create(
            [
                node("jingya-dev-tpu-v6e1", state="STOPPED"),
                node(f"amoran-tpu-dev-flex-{ZONE}", state="TERMINATED"),
                node(f"other-{NAME}"),
            ]
        )
        self.command.assert_called_once_with(
            f"gcloud alpha compute tpus tpu-vm create {NAME} --zone {ZONE}"
            " --accelerator-type=v6e-4 --version=v2-alpha-tpuv6e"
        )
        self.save.assert_called_once_with({NAME: {"type": "v6e-4", "zone": ZONE}})
        self.install.assert_called_once()

    def test_create_skips_exact_name_even_when_terminated(self):
        result = self.invoke_create([node(NAME, state="TERMINATED")])
        self.assertIn("already exists", result.output)
        self.command.assert_not_called()
        self.save.assert_not_called()
        self.install.assert_not_called()

    def test_create_in_empty_zone(self):
        self.invoke_create([])
        self.command.assert_called_once()
        self.install.assert_called_once()

    def test_state_selects_exact_name(self):
        self.output.return_value = json.dumps(
            [
                node(f"other-{NAME}", state="STOPPED"),
                node(NAME),
            ]
        )
        self.assertEqual(get_tpu.get_state(NAME, ZONE), "READY")

    def test_state_reports_missing_for_suffix_match_only(self):
        self.output.return_value = json.dumps([node(f"other-{NAME}")])
        self.assertEqual(get_tpu.get_state(NAME, ZONE), "NOT FOUND")

    def test_ip_selects_exact_name(self):
        self.output.return_value = json.dumps(
            [
                node(f"other-{NAME}", ip="192.0.2.2"),
                node(NAME),
            ]
        )
        self.assertEqual(get_tpu.get_ext_ip(NAME, ZONE), "192.0.2.1")


if __name__ == "__main__":
    unittest.main()
