
import numpy as np
X_GTEx = np.load('GTEx_X_float64.npy')
def write_g_file_prefix(g):
    g.write('\\documentclass{article}\n\
    \\usepackage{tikz}\n\
    \\usepackage{pgfplots}\n\
    \\usepackage{textcomp}\n\
    \\usepackage{array}\n\
    \\usepackage{tabu}\n\
    \\usepackage{numprint}\n\
    \\begin{document}')
def write_t_file_prefix(t):
    t.write('\\documentclass{article}\n\
    \\usepackage{tikz}\n\
    \\usepackage{pgfplots}\n\
    \\usepackage{textcomp}\n\
    \\usepackage{array}\n\
    \\usepackage{tabu}\n\
    \\usepackage{numprint}\n\
    \\begin{document}')
def write_g_prefix(g, data# , noise_ratio
):
    max_ = max(data)
    min_ = min(data)

    diff = max_ - min_
    g.write('\n\n\\begin{tikzpicture}\n')
    g.write('\\begin{axis}[\n')
    #g.write('title={Sample size of training data plotted against accuracy at '+str(noise_ratio)+' noise ratio},\n')
    g.write('xlabel={Number of Samples},\n')
    g.write('ylabel={Mean Square Error between all original and denoised samples},\n')
    g.write('xmin=0, xmax=3000,\n')
    g.write('ymin='+str(min_)+', ymax='+str(max_)+',\n')
    g.write('xtick={},\n')
    g.write('ytick={'+str(min_)+','+str(min_+1*diff)+','+str(min_+2*diff)+','+str(min_+3*diff)+','+str(min_+4*diff)+','+str(min_+5*diff)+','+str(min_+6*diff)+','+str(min_+7*diff)+','+str(min_+8*diff)+','+str(min_+9*diff)+','+str(min_+10*diff)+'},\n')
    g.write('legend pos=north west,\n')
    g.write('ymajorgrids=true,\n')
    g.write('grid style=dashed,\n')
    g.write(']\n\n')
    g.write('\\addplot[\n')
    g.write('color=blue,\n')
    g.write('mark=square,\n')
    g.write(']\n')
    g.write('coordinates {\n\n')

def write_t_prefix(t):
    t.write('\\npdecimalsign{.}\n')
    t.write('\\nprounddigits{2}\n')
    t.write('\\begin{tabu} to 0.8\\textwidth { | X[l] | X[r] |}\n')
    t.write('\\hline\n')
    t.write('samples &  MSE\\\\\n')
    t.write('\\hline\n')

def write_g_suffix(g):
    g.write('    };\n')
    g.write('\\end{axis}\n')
    g.write('\\end{tikzpicture}\n')

def write_t_suffix(t):
    t.write('\\end{tabu}\n')
    t.write('\\npnoround\n')

def write_g_file_suffix(g):
    g.write('\n\
\\end{document}\n')
    
def write_t_file_suffix(t):
    t.write('\\end{document}')

def write_g_tex_file(MSE, filename, tag):
    g = open(filename, tag)
    write_g_file_prefix(g)
    for r in range(0,len(MSE)):
        noise_factor = r * 0.1
        write_g_prefix(g, MSE[r], noise_factor)  
        for s in range(1,11):
            samples_ratio = s * 0.1
            samples = int((samples_ratio)*len(X_GTEx))
            g.write('('+str(samples)+', '+str(MSE[r][s-1])+')\n')
        write_g_suffix(g)
    write_g_file_suffix(g)
    g.close()
   

def write_t_tex_file(MSE, filename, tag):

    t = open(filename, tag)
    write_t_file_prefix(t)

    for r in range(0,len(MSE)):
        noise_factor = r * 0.1 
        write_t_prefix(t) # calculate ticks. 
        for s in range(1,11):
            samples_ratio = s * 0.1
            samples = int((samples_ratio)*len(X_GTEx))
            t.write(str(samples)+' & '+str(MSE[r][s-1]) +'\\'+'\\' + '\n'+'\hline\n')
        write_t_suffix(t)
    write_t_file_suffix(t)    
    t.close()


def write_g_tex_file_1D(MSE, filename, tag):
    g = open(filename, tag)
    write_g_file_prefix(g)
    write_g_prefix(g, MSE)
    for r in range(len(MSE)):
        noise_factor = r * 0.1
        g.write('('+str(noise_factor)+', '+str(MSE[r])+')\n')
    write_g_suffix(g)
    write_g_file_suffix(g)
    g.close()
   

def write_t_tex_file_1D(MSE, filename, tag):
    t = open(filename, tag)
    write_t_file_prefix(t)
    write_t_prefix(t)
    for r in range(len(MSE)):
        noise_factor = r * 0.1
        t.write(str(noise_factor)+' & '+str(MSE[r]) +'\\'+'\\' + '\n'+'\hline\n')
    write_t_suffix(t)
    write_t_file_suffix(t)
    t.close()



    
