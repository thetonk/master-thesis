class SlurmTimeLimitException(Exception):
    def __str__(self):
        return "SLURM time out! Job is getting killed!"


def handle_slurm_exception(sig_num, frame):
    raise SlurmTimeLimitException()