"""Small offscreen GLES helpers for isolated experiments, never HDMI scanout."""
import os
os.environ['PYOPENGL_PLATFORM']='egl'
os.environ['EGL_PLATFORM']='surfaceless'
import ctypes as C
from OpenGL.EGL import *
from OpenGL.GLES3 import *


def context():
    display=eglGetDisplay(EGL_DEFAULT_DISPLAY)
    major,minor=EGLint(),EGLint()
    eglInitialize(display,major,minor)
    attrs=(EGLint*13)(EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,0x40,
                      EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE)
    configs=(EGLConfig*1)();count=EGLint()
    eglChooseConfig(display,attrs,configs,1,count)
    assert count.value
    eglBindAPI(EGL_OPENGL_ES_API)
    ctx=eglCreateContext(display,configs[0],EGL_NO_CONTEXT,(EGLint*3)(EGL_CONTEXT_CLIENT_VERSION,3,EGL_NONE))
    surface=eglCreatePbufferSurface(display,configs[0],(EGLint*5)(EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE))
    eglMakeCurrent(display,surface,surface,ctx)
    renderer=C.string_at(glGetString(GL_RENDERER)).decode()
    print('RENDERER',renderer,C.string_at(glGetString(GL_VERSION)).decode(),flush=True)
    glBindVertexArray(glGenVertexArrays(1));glDisable(GL_DITHER)
    return display,ctx,surface,renderer


def program(fragment):
    vertex='''#version 300 es
    const vec2 p[3]=vec2[3](vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.));
    void main(){gl_Position=vec4(p[gl_VertexID],0.,1.);}
    '''
    if fragment.startswith('#version 310 es'):
        vertex=vertex.replace('#version 300 es','#version 310 es')
    shaders=[]
    for kind,source in [(GL_VERTEX_SHADER,vertex),(GL_FRAGMENT_SHADER,fragment)]:
        shader=glCreateShader(kind);glShaderSource(shader,source);glCompileShader(shader)
        if not glGetShaderiv(shader,GL_COMPILE_STATUS):raise RuntimeError(glGetShaderInfoLog(shader))
        shaders.append(shader)
    result=glCreateProgram()
    for shader in shaders:glAttachShader(result,shader)
    glLinkProgram(result)
    if not glGetProgramiv(result,GL_LINK_STATUS):raise RuntimeError(glGetProgramInfoLog(result))
    for shader in shaders:glDeleteShader(shader)
    return result


def texture(unit,array,internal,fmt,typ,width=None,height=None,immutable=False):
    result=glGenTextures(1);glActiveTexture(GL_TEXTURE0+unit);glBindTexture(GL_TEXTURE_2D,result)
    for key,value in [(GL_TEXTURE_MIN_FILTER,GL_NEAREST),(GL_TEXTURE_MAG_FILTER,GL_NEAREST),
                      (GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE),(GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE)]:
        glTexParameteri(GL_TEXTURE_2D,key,value)
    if array is not None:height,width=array.shape[:2]
    if immutable:
        glTexStorage2D(GL_TEXTURE_2D,1,internal,width,height)
        if array is not None:glTexSubImage2D(GL_TEXTURE_2D,0,0,0,width,height,fmt,typ,array.ctypes.data_as(C.c_void_p))
    else:
        glTexImage2D(GL_TEXTURE_2D,0,internal,width,height,0,fmt,typ,
                     None if array is None else array.ctypes.data_as(C.c_void_p))
    return result


def framebuffer(tex):
    result=glGenFramebuffers(1);glBindFramebuffer(GL_FRAMEBUFFER,result)
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tex,0)
    assert glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE
    return result
