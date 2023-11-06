; SPDX-FileCopyrightText: Idiap Research Institute
; SPDX-FileContributor: Enno Hermann <enno.hermann@idiap.ch>
;
; SPDX-License-Identifier: GPL-3.0-only

(defun spe/ox-ipynb-export-to-ipynb-file ()
  "Export current buffer using `ox-ipynb-export-to-ipynb-file':
- first remove the results
- save to ../jupyter-notebooks/*.ipynb
- don't open the .ipynb file afterwards"
  (let ((ox-ipynb-preprocess-hook '((lambda ()
				      (org-babel-map-src-blocks nil
					(org-babel-remove-result))))))
    (setq-local export-file-name
                (file-name-concat "../jupyter-notebooks"
                                  (concat (file-name-base) ".ipynb")))
    (kill-buffer
     (find-file-noselect
      (ox-ipynb-export-to-ipynb-file)))))
