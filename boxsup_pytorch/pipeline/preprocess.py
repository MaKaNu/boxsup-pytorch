from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy

from boxsup_pytorch.config import GLOBAL_CONFIG
from boxsup_pytorch.utils.check import check_exists_msg, check_init_msg
from boxsup_pytorch.utils.common import counter, print_progress_bar


@dataclass
class PreProcessMCG():
    in_masks_location: Optional[Path] = None
    out_masks_location: Optional[Path] = None

    def update(self):
        assert self.in_masks_location is not None, check_init_msg()

        self.out_masks_location = self.in_masks_location.parent / "MCG_processed/"
        not_processed_files = self._get_unprocessed_files()
        self._load_and_process_matfiles(not_processed_files)

    def set_inputs(self,  inputs: Dict[str, Path]):
        assert "masks_location" in inputs.keys(), check_exists_msg("masks_location")

        self.in_masks_location = inputs['masks_location']

    def get_outputs(self):
        return {'labelmask': self.out_masks_location}

    @counter
    def _get_unprocessed_files(self) -> List[str]:
        """Create a list of file stems which are not processed.

        After checking if the in-/outputs of the Pipeline Process are set,
        the method creates the masks location if the location not yet exists.
        Depending on the config 'rerun_process' boolean, the npz-files in the target
        location will be deleted.

        Returns:
            List[str]: Depending on the comparison between the files in the
                       'self.in_mask_location' and the files in the 
                       'self.out_masks_location'  a List of file stems
        """
        assert self.in_masks_location is not None, check_init_msg()
        assert self.out_masks_location is not None, check_init_msg()

        if not self.out_masks_location.exists():
            self.out_masks_location.mkdir()

        files = [file.stem for file in self.in_masks_location.glob("*.mat")]
        processed_files = [file.stem for file in self.out_masks_location.glob("*.npz")]

        if GLOBAL_CONFIG.rerun_process:
            for file in processed_files:
                (self.out_masks_location / (file + '.npz')).unlink()

        not_processed_files = [file for file in files if file not in processed_files]
        return not_processed_files

    def _load_and_process_matfiles(self, list_of_files) -> None:
        """Iterate over all MATLAB files and generate the candidate masks.

        The method extracts the 'scores', 'superpixels' and 'labels' from the MATLAB-file dict.
        Based on the 'scores' the TOP N candidates are choosen, while N is defined in the config.
        The mask creation is based on the 'superpixels' 2D array which is compared to the 'labels'.
        The stacked masked are saved as compressed numpy file at the 'self.out_masks_location'. 

        Args:
            list_of_files (List[str]): The list, which is created by the '_get_unprocessed_files.
        """
        def _process_matfiles():
            top_n = GLOBAL_CONFIG.mcg_num_candidates
            top_n_idx = np.argpartition(scores.squeeze(), -top_n)[-top_n:]
            masks = []
            for idx in top_n_idx:
                mask = np.isin(superpixels, labels[idx, 0])
                mask = mask.reshape(superpixels.shape)
                mask = mask.astype(np.float32)
                masks.append(mask)
            masks = np.stack(masks)
            save_path = self.out_masks_location / (mat_file + ".npz")
            np.savez_compressed(save_path, masks=masks)

        assert self._get_unprocessed_files.invocations, "'self._get_unprocessed_files' not called!"

        total_steps = len(list_of_files)
        for idx, mat_file in enumerate(list_of_files):
            mcg_mat = scipy.io.loadmat(self.in_masks_location / (mat_file + ".mat"))
            scores = mcg_mat["scores"]
            superpixels = mcg_mat["superpixels"]
            labels = mcg_mat["labels"]
            _process_matfiles()
            print_progress_bar(idx, total_steps, title=f"File: {mat_file}")


if __name__ == "__main__":
    process_instance = PreProcessMCG()
    # set input
    inputs = {"masks_location": GLOBAL_CONFIG.mcg_path}
    process_instance.set_inputs(inputs)
    # update
    process_instance.update()
