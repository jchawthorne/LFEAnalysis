from . import read_catalogues
from . import radpatterns
from . import seisproc
from . import _lfe_initiation
from . import _seismic_data_management
from . import _lfe_detections
from . import _lfe_arrivals
from . import _lfe_download_data
from . import _lfe_stacking
from . import _lfe_grouping
from . import _lfe_xc
from . import _lfe_stations_orientations
from . import _lfe_scaling
from . import _lfe_results_saving
from . import _lfe_radpatterns


# initiate an overall class for the LFE analysis
class lfeanalyse(_lfe_initiation.lfeanalyse,
                 _seismic_data_management.lfeanalyse,
                 _lfe_detections.lfeanalyse,
                 _lfe_arrivals.lfeanalyse,
                 _lfe_download_data.lfeanalyse,
                 _lfe_stacking.lfeanalyse,
                 _lfe_grouping.lfeanalyse,
                 _lfe_xc.lfeanalyse,
                 _lfe_stations_orientations.lfeanalyse,
                 _lfe_scaling.lfeanalyse,
                 _lfe_results_saving.lfeanalyse,
                 _lfe_radpatterns.lfeanalyse):

    def trashfunction(self):
        pass



from importlib import reload
def reload_all(mdl):
    reload(mdl)
    for vl in mdl.__dict__.values():
        try: 
            reload(vl)
        except:
            pass
    reload(mdl)
