"""Sparse, antialiased GPX pixels; only uniforms change during the reveal."""
import ctypes as C
from OpenGL.GLES3 import *
from .gpu import program

VERTEX='''#version 300 es
precision highp float;
layout(location=0) in vec4 point;
out float phase;out float coverage;
void main(){gl_Position=vec4(point.xy,0.,1.);gl_PointSize=1.;phase=point.z;coverage=point.w;}
'''
FRAGMENT='''#version 300 es
precision highp float;
in float phase;in float coverage;
uniform float reveal;uniform vec3 ink;
out vec4 color;
void main(){float a=coverage*(1.-smoothstep(reveal-.008,reveal,phase));if(a<.001)discard;color=vec4(ink,a);}
'''


class TrackRenderer:
    def __init__(self):
        self.program=program(FRAGMENT,VERTEX);self.vao=glGenVertexArrays(1);self.buffer=glGenBuffers(1);self.vertices=None;self.count=0
        self.reveal=glGetUniformLocation(self.program,'reveal');self.ink=glGetUniformLocation(self.program,'ink')

    def draw(self,state,target,width,height):
        vertices=state['vertices']
        glBindVertexArray(self.vao);glBindBuffer(GL_ARRAY_BUFFER,self.buffer)
        if self.vertices is not vertices:
            glBufferData(GL_ARRAY_BUFFER,vertices.nbytes,vertices,GL_STATIC_DRAW);self.vertices=vertices;self.count=len(vertices)
            glEnableVertexAttribArray(0);glVertexAttribPointer(0,4,GL_FLOAT,False,16,C.c_void_p(0))
        glBindFramebuffer(GL_FRAMEBUFFER,target);glViewport(0,0,width,height);glUseProgram(self.program)
        glUniform1f(self.reveal,state['progress']*1.008)
        ink=state['color'];glUniform3f(self.ink,*[int(ink[i:i+2],16)/255 for i in (1,3,5)])
        glEnable(GL_BLEND);glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);glDrawArrays(GL_POINTS,0,self.count);glDisable(GL_BLEND)
        glBindVertexArray(0)

    def close(self):
        glDeleteBuffers(1,[self.buffer]);glDeleteVertexArrays(1,[self.vao]);glDeleteProgram(self.program)
