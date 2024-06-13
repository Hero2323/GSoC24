import re

def license_parser(prompt_result):
  """Parses the prompt result for software licenses and SPDX identifiers."""

  licenses = []
  spdx_ids = []

  license_matches = re.findall(
      r"(?P<license>\b[A-Z][a-zA-Z]+\sLicense\b)"  # Match license names (e.g., MIT License)
      r"\s*,\s*"                                  # Optional comma and whitespace
      r"(?P<spdx>SPDX-License-Identifier:\s*[A-Za-z-]+)",  # Match SPDX identifier
      prompt_result, 
      re.IGNORECASE
  )
  for match in license_matches:
      license_name, spdx_id = match
      licenses.append(license_name)
      spdx_ids.append(spdx_id.split(": ")[1])  # Extract the SPDX identifier itself

  if not licenses:
    return {"Licenses": [], "SPDX-IDs": []}
  else:
    return {"Licenses": licenses, "SPDX-IDs": spdx_ids}    