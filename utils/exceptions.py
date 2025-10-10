# Copyright (C) 2025 Spyridon Baltsas
# This file is part of the research project "Cyberattack detection on network level using state-of-the-art deep learning models"
# Licensed under the GNU General Public License v3.0 (GPLv3)
# See the LICENSE file in the project root for full license text.

class SlurmTimeLimitException(Exception):
    def __str__(self):
        return "SLURM time out! Job is getting killed!"


class InvalidArgumentException(Exception):
    pass


def handle_slurm_exception(sig_num, frame):
    raise SlurmTimeLimitException()