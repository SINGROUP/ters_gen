import argparse
from pathlib import Path

def read_log(read_type, log_file):
    """
    Can read log files and return the desired information.

    Arguments:
        read_type: int -- 0 for checking unfinished tasks, 1 for checking tasks with errors

    Returns:
        set -- set of the name of .fchk files that are unfinished or have errors
    """

    def check_unfinished_tasks(log_file):
        """
        Returns started but not finished tasks from the given log file.
        """
        with open(log_file, 'r') as file:
            logs = file.readlines()
        
        start_tasks = set()
        finish_tasks = set()
        
        for log in logs:
            if 'Started processing file:' in log:
                start_tasks.add(log.split('Started processing file: ')[1].strip())
            elif 'Finished processing file:' in log:
                finish_tasks.add(log.split('Finished processing file: ')[1].strip())
        
        unfinished_tasks = start_tasks - finish_tasks
        print(unfinished_tasks)

        return unfinished_tasks


    def check_errors(log_file):
        """
        Returns tasks that have errors from the given log file.
        """
        with open(log_file, 'r') as file:
            logs = file.readlines()
        
        error_tasks = set()
        
        for log in logs:
            if 'Error processing file:' in log:
                error_tasks.add(log.split('Error processing file: ')[1].strip())
        print(error_tasks)

        return error_tasks
    
    if read_type == 0:
        return check_unfinished_tasks(log_file)
    elif read_type == 1:
        return check_errors(log_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('read_type', type=int, help='0 for checking unfinished tasks, 1 for checking tasks with errors')
    parser.add_argument('log_file', type=str, help='The log file to read')
    args = parser.parse_args()

    log_file = Path(args.log_file)

    read_log(args.read_type, log_file)