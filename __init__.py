from . import read_catalogues
from . import _lfe_initiation
from . import _seismic_data_management
from . import _lfe_detections
from . import _lfe_arrivals
from . import _lfe_download_data
from . import _lfe_stacking


# initiate an overall class for the LFE analysis
class lfeanalyse(_lfe_initiation.lfeanalyse,
                 _seismic_data_management.lfeanalyse,
                 _lfe_detections.lfeanalyse,
                 _lfe_arrivals.lfeanalyse,
                 _lfe_download_data.lfeanalyse,
                 _lfe_stacking.lfeanalyse):

    def trashfunction(self):
        pass



from importlib import reload
def reload_all(mdl):
    for vl in mdl.__dict__.values():
        try: 
            reload(vl)
        except:
            pass
    reload(mdl)
