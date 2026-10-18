import os
import sys
from typing import Tuple

# Off-screen OpenGL context backend: EGL renders on the GPU with no window (Linux); Windows and macOS have no EGL,
# so a hidden GLFW window provides the context there. Override with RTV_GL_BACKEND=egl|glfw.
GL_BACKEND = os.environ.get('RTV_GL_BACKEND') or ('egl' if sys.platform.startswith('linux') else 'glfw')
if GL_BACKEND == 'egl':
    # must be set before the first OpenGL import anywhere in the process
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import ctypes

import numpy as np
from OpenGL import error
from OpenGL.GL import *


class _EGLContext:
    def __init__(self, surface_size):
        import OpenGL.EGL as egl
        from OpenGL.raw.EGL.EXT.platform_device import EGL_PLATFORM_DEVICE_EXT
        from OpenGL.EGL.EXT.device_base import egl_get_devices
        self.egl = egl

        display = egl.EGL_NO_DISPLAY
        for device in egl_get_devices():
            candidate = egl.eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, device, None)
            if candidate != egl.EGL_NO_DISPLAY and egl.eglGetError() == egl.EGL_SUCCESS:
                # `eglInitialize` may or may not raise on failure depending on how PyOpenGL is configured,
                # so catch GLError and also check eglGetError()
                try:
                    initialized = egl.eglInitialize(candidate, None, None)
                except error.GLError:
                    continue
                if initialized == egl.EGL_TRUE and egl.eglGetError() == egl.EGL_SUCCESS:
                    display = candidate
                    break
        if display == egl.EGL_NO_DISPLAY:
            raise RuntimeError('Cannot initialize a headless EGL display.')

        major, minor = egl.EGLint(), egl.EGLint()
        egl.eglInitialize(display, ctypes.pointer(major), ctypes.pointer(minor))
        config_attribs = [
            egl.EGL_SURFACE_TYPE, egl.EGL_PBUFFER_BIT, egl.EGL_BLUE_SIZE, 8,
            egl.EGL_GREEN_SIZE, 8, egl.EGL_RED_SIZE, 8, egl.EGL_DEPTH_SIZE, 24,
            egl.EGL_RENDERABLE_TYPE, egl.EGL_OPENGL_BIT, egl.EGL_NONE
        ]
        config_attribs = (egl.EGLint * len(config_attribs))(*config_attribs)
        num_configs = egl.EGLint()
        egl_cfg = egl.EGLConfig()
        egl.eglChooseConfig(display, config_attribs, ctypes.pointer(egl_cfg), 1, ctypes.pointer(num_configs))

        width, height = surface_size
        pbuffer_attribs = [egl.EGL_WIDTH, width, egl.EGL_HEIGHT, height, egl.EGL_NONE]
        pbuffer_attribs = (egl.EGLint * len(pbuffer_attribs))(*pbuffer_attribs)
        self.display = display
        self.surface = egl.eglCreatePbufferSurface(display, egl_cfg, pbuffer_attribs)
        egl.eglBindAPI(egl.EGL_OPENGL_API)
        self.context = egl.eglCreateContext(display, egl_cfg, egl.EGL_NO_CONTEXT, None)
        self.make_current()

    def make_current(self):
        self.egl.eglMakeCurrent(self.display, self.surface, self.surface, self.context)

    def destroy(self):
        self.egl.eglDestroySurface(self.display, self.surface)
        self.egl.eglDestroyContext(self.display, self.context)


class _GLFWContext:
    def __init__(self, surface_size):
        try:
            import glfw
        except ImportError:
            raise RuntimeError('OpenGL rendering on %s needs the glfw package: pip install glfw' % sys.platform)
        self.glfw = glfw
        if not glfw.init():
            raise RuntimeError('glfw.init() failed; check the GPU driver supports OpenGL 3.3')
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # never shown: rendering goes to an off-screen framebuffer
        # the GL viewport starts at the size of the first surface, so the hidden window must match the render size
        self.window = glfw.create_window(surface_size[0], surface_size[1], 'rtv-offscreen', None, None)
        if not self.window:
            raise RuntimeError('Could not create a hidden OpenGL window with glfw')
        self.make_current()
        print('OpenGL renderer: %s (%s)' % (glGetString(GL_RENDERER).decode(), glGetString(GL_VERSION).decode()))

    def make_current(self):
        self.glfw.make_context_current(self.window)

    def destroy(self):
        self.glfw.destroy_window(self.window)


def create_opengl_context(surface_size: Tuple[int, int]):
    """Create an off-screen OpenGL context and make it current.

    Args:
        surface_size: (width, height); rendering itself goes to a framebuffer object of the caller's size.

    Returns an object with make_current() and destroy().
    """
    return _EGLContext(surface_size) if GL_BACKEND == 'egl' else _GLFWContext(surface_size)


def init_frame_buffer(frame_image: np.ndarray, num_samples: int = 16):
    """Create an FBO and assign a texture buffer to it for the purpose of offscreen rendering to the texture buffer
    """
    samples = min(num_samples, GL_SAMPLES - 1)

    width = frame_image.shape[1]
    height = frame_image.shape[0]

    if frame_image.shape[-1] == 3:
        texture_type = GL_RGB
        internal_texture_type = GL_RGB8
    elif frame_image.shape[-1] == 4:
        texture_type = GL_RGBA
        internal_texture_type = GL_RGBA8

    render_texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, render_texture)
    glTexImage2D(GL_TEXTURE_2D, 0, internal_texture_type, width,
                 height, 0, texture_type, GL_UNSIGNED_BYTE, None)

    glBindTexture(GL_TEXTURE_2D, 0)

    render_frame_object = glGenFramebuffers(1)

    glBindFramebuffer(GL_FRAMEBUFFER, render_frame_object)

    # Create color render buffer
    color_buffer = glGenRenderbuffers(1)

    glBindRenderbuffer(GL_RENDERBUFFER, color_buffer)
    glRenderbufferStorage(
        GL_RENDERBUFFER, internal_texture_type, width, height)
    glFramebufferRenderbuffer(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color_buffer)

    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, render_texture, 0)

    depth_buffer = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height)
    glBindRenderbuffer(GL_RENDERBUFFER,0)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depth_buffer)

    glBindRenderbuffer(GL_RENDERBUFFER, 0)

    # Sees if your GPU can handle the FBO configuration defined above
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(
            "Framebuffer binding failed, probably because your GPU does not support this FBO configuration.")

    glBindFramebuffer(GL_FRAMEBUFFER, 0)



    return render_frame_object
