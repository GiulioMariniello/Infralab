"""Plugin degli asset type: ogni tipo registra profili componenti e stati limite."""
from infralab.assets.base import AssetTypeSpec  # noqa: F401
from infralab.assets.registry import (  # noqa: F401
    available_asset_types,
    get_asset_type,
    register_asset_type,
)

# Registrazione dei tipi built-in (import con side effect di registrazione).
import infralab.assets.bridges  # noqa: E402,F401
import infralab.assets.retaining_walls  # noqa: E402,F401
import infralab.assets.wind_turbines  # noqa: E402,F401
import infralab.assets.buildings  # noqa: E402,F401
import infralab.assets.solar_trackers  # noqa: E402,F401
