from pathlib import Path
import numpy as np
from src.reports.guidance_report import write_guidance


def test_empty_guidance_has_five_pages_and_sop(tmp_path):
    result=write_guidance([],[],{},[],[],[],{'warning_threshold':.2,'high_threshold':.4},tmp_path,100,{})
    assert result['instances']==0
    for name in ['index.html','01_overlabel.html','02_gray_and_clusters.html','03_mask_quality.html','04_missing.html','05_proto.html','sop.html']:
        assert (tmp_path/name).is_file()
    assert not list(tmp_path.rglob('*.csv'))
