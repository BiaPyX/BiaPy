"""
Reusable, workflow-specific helpers for the BiaPy engine.

This subpackage collects logic that belongs to a single workflow but is large or self-contained
enough to keep out of the main workflow module (which would otherwise grow unwieldy). Each helper is
typically exposed as a mixin the workflow inherits, so the code keeps natural ``self`` access to the
workflow state while living in its own file.
"""
