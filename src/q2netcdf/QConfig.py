#
# Decode QFile config record into a dictionary
#
# Feb-2025, Pat Welch, pat@mousebrains.com

import re
import numpy as np
import json
try:
    from QVersion import QVersion
except:
    from q2netcdf.QVersion import QVersion

class QConfig:
    def __init__(self, config:str, version:QVersion) -> None:
        self.__config = config
        self.__version = version
        self.__dict = None

    def __repr__(self) -> str:
        config = self.config()
        msg = []
        for key in sorted(config):
            msg.append(f"{key} -> {config[key]}")
        return "\n".join(msg)

    def __parseValue(self, val:str):
        matches = re.match(r"^\[(.*)\]$", val)
        if matches:
            fields = []
            for field in matches[1].split(","):
                fields.append(self.__parseValue(field.strip()))
            return np.array(fields)

        matches = re.match(r"^[+-]?\d+$", val)
        if matches: return int(val)

        matches = re.match(r"^[+-]?\d+[.]\d*(|[Ee][+-]?\d+)$", val)
        if matches: return float(val)

        matches = re.match(r'^"(.*)"$', val)
        if matches: return matches[1]

        matches = re.match(r"^true$", val)
        if matches: return True

        matches = re.match(r"^false$", val)
        if matches: return False

        return val

    def __splitConfigV12(self) -> None:
        self.__dict = dict()
        for line in self.__config.split(b"\n"):
            try:
                line = str(line, "utf-8").strip()
                matches = re.match(r'^"(.*)" => (.*)$', line)
                if matches:
                    self.__dict[matches[1]] = self.__parseValue(matches[2])
            except:
                pass

    def __splitConfigv13(self) -> None:
        self.__dict = json.loads(self.__config)

    def __len__(self) -> int:
        return len(self.__config)

    def size(self) -> int:
        return len(self)

    def raw(self) -> str:
        return self.__config

    def config(self) -> dict:
        if self.__dict is None:
            if self.__version.isV12():
                self.__splitConfigV12()
            else:
                self.__splitConfigv13()
        return self.__dict
