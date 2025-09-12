class SlurmTimeLimitException(Exception):
    def __str__(self):
        return "SLURM time out! Job is getting killed!"


class InvalidArgumentException(Exception):
    pass


def handle_slurm_exception(sig_num, frame):
    raise SlurmTimeLimitException()