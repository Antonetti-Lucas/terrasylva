"""Fail the build if pytest skipped something it should not have.

Usage: python check_skips.py report.xml [reason ...]

With no reasons, no test may be skipped. With reasons, at least one skip must match
them and no skip may have another message. Outside Windows, the layer hash test may
also be skipped.
"""
import sys
import xml.etree.ElementTree as ET

HASH_TEST = "tests.test_layer_hashes::test_layers_match_stored_hashes"
HASH_REASON = "layer hashes are stored for Windows only"

report, *allowed = sys.argv[1:]
cases = list(ET.parse(report).iter("testcase"))
skips = [(f"{case.get('classname')}::{case.get('name')}", skip.get("message", ""))
         for case in cases for skip in case.findall("skipped")]


def given(why):
    return any(a in why for a in allowed)


def hash_skip(name, why):
    return sys.platform != "win32" and name == HASH_TEST and why == HASH_REASON


unexpected = [name for name, why in skips if not (given(why) or hash_skip(name, why))]

for name, why in skips:
    print(f"skipped {name}: {why}" + ("  <- unexpected" if name in unexpected else ""))
print(f"{len(cases)} tests, {len(skips)} skipped, {len(unexpected)} unexpected")

if not cases or unexpected or (allowed and not any(given(why) for _, why in skips)):
    sys.exit(1)
