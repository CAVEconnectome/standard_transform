from .datasets import (
    # unified resolution= API
    minnie_transform,
    v1dd_transform,
    minnie_streamline,
    v1dd_streamline,
    available_versions,
    # dataset bundles
    v1dd_ds,
    minnie_ds,
    # identities
    identity_transform,
    identity_streamline,
    # deprecated _nm/_vx aliases (superseded by the resolution= API)
    minnie_transform_nm,
    minnie_transform_vx,
    v1dd_transform_nm,
    v1dd_transform_vx,
    v1dd_streamline_nm,
    v1dd_streamline_vx,
)
from .streamlines import (
    Streamline,
    StreamlineField,
    streamline_field_from_paths,
)

__version__ = "2.0.0"
