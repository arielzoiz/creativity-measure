"""Model backends for samplers that drive a pretrained generative model they do not own.

Every module here is an **optional** import: the third-party stack it bridges to (JAX/Flax, a vendor
checkpoint, a cloned upstream repo) must never become a dependency of ``creativity_measure`` proper.
Import the concrete backend directly, e.g.::

    from creativity_measure.backends.diamond_maps_jax import DiamondMapsBackend

which raises a clear ``ImportError`` when the extra stack is missing, exactly as
``generators/flux.py`` does for ``diffusers``.
"""
