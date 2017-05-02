#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import commands.content
import output
from spiderpig import run_cli
from config import get_argument_parser


if __name__ == '__main__':
    run_cli(
        namespaced_command_packages={
            'content': commands.content,
        },
        argument_parser=get_argument_parser(),
        setup_functions=[output.init_plotting]
    )
