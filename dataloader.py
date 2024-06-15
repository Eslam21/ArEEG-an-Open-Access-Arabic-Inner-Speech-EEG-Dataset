from Utilities import process_eeg
import os
from concurrent.futures import ProcessPoolExecutor

def LoadAllSubjectsCSV(full_dataset_path: str = 'CSVData'):
    """
    Load all subject data from CSV files.

    Parameters: dataset folder path
    """
    results = []
    with ProcessPoolExecutor(4) as executor:
        for subject in os.listdir(full_dataset_path):
            results.append(executor.submit(process_eeg, subject_folder=os.path.join(full_dataset_path,subject)))

    for result in results:
        yield result.result()

def LoadAllSubjectsNPY(full_dataset_path: str = 'NPYData'):
    """
    Load all subject data from NPY files.

    Parameters: dataset folder path
    """
    results = []
    with ProcessPoolExecutor(4) as executor:
        for subject in os.listdir(full_dataset_path):
            results.append(executor.submit(process_eeg, subject_folder=os.path.join(full_dataset_path,subject)))
    return [result.result() for result in results]

if __name__ == "__main__":
    for x,y in zip(LoadAllSubjectsCSV('CSVData'), LoadAllSubjectsNPY('NPYData')):
        assert (x[0] == y[0]).all()
        assert (x[1] == y[1]).all()
        print(x[0].shape, y[0].shape , x[1].shape, y[1].shape)