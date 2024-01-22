import subprocess


def start_subprocess(command, print_command=False, pipe_command=None):
    if print_command:
        print(" ".join(command) + "\n")
    if pipe_command:
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output = subprocess.check_output(pipe_command, stdin=process.stdout)
    else:
        process = subprocess.Popen(command)

    return process.wait()
