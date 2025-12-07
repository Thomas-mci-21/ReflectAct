"""Utility functions for task handling (aligned with MPO)."""

def process_ob(ob):
    """Process observation (aligned with MPO)."""
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob

