import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Importando as bibliotecas necessarias
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import math


#Funcao responsável por ler o arquivo
def ler_arquivo(caminho_arquivo):
    """
    Lê um arquivo de dados contendo informações do perfil aerodinâmico e armazena os dados em um DataFrame do Pandas.
    
    Parâmetros:
    caminho_arquivo (str): O caminho para o arquivo de dados.

    Retorno:
    pd.DataFrame: Um DataFrame contendo os dados do perfil aerodinâmico com colunas 'x', 'y' e 'Cp'.
    """
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    data = pd.read_csv(caminho_arquivo, delim_whitespace=True, names=["x", "y", "Cp"])
    data['-Cp'] = -data['Cp']

    return data

#Funcao responsável por plotar o grafico dos dados fornecidos
def plotar_grafico(data):
    """
    Plota um gráfico do perfil do aerofólio com Cp em função de x.

    Parâmetros:
    data (pd.DataFrame): O DataFrame com os dados do perfil aerodinâmico.

    Retorno:
    None
    """
    positivos = data[data['y'] > 0]
    negativos = data[data['y'] < 0]

    plt.plot(positivos['x'], positivos['-Cp'], linestyle='-', color='blue', label='C_p,u')
    plt.plot(negativos['x'], negativos['-Cp'], linestyle='-', color='red', label='C_p,l')
    plt.title('Gráfico de Linha Contínua de -Cp em Função de x')
    plt.xlabel('x')
    plt.ylabel('-Cp')
    plt.grid(True)
    plt.legend()
    plt.show()

#Calcula o Mach critico
def calcular_mach_critico(data, gama=1.4):
    """
    Calcula o Mach crítico com base nos dados de Cp, usando uma equação analítica.

    Parâmetros:
    data (pd.DataFrame): O DataFrame com os dados do perfil aerodinâmico.
    gama (float): A razão de calor (padrão: 1.4).

    Retorno:
    pd.DataFrame: Um DataFrame com os valores de Mach crítico calculados.
    """
    cp_min = data['Cp'].min()
    relacao_gama = (gama - 1) / 2

    df = pd.DataFrame(columns=['mcr', 'cp', 'cp_2'])
    mcr = np.arange(0.5, 0.8, 0.0000001)
    cp_list = []
    cp_2_list = alpha = np.radians(3)

    for i in mcr:
        cp = cp_min / (np.sqrt(1 - i**2))
        cp_2 = (2 / (gama * ((i**2))) ) * (( ( (1 + relacao_gama * (i**2) ) / (1 + relacao_gama) ) ** (gama/(gama - 1)) ) - 1)
        cp_list.append(cp)
        cp_2_list.append(cp_2)

    df['mcr'] = mcr
    df['cp'] = cp_list
    df['cp_2'] = cp_2_list
    df['diff'] = np.abs(df['cp'] - df['cp_2'])

    return df


#Faz a iteracao para encontrar o valor de mach critico 
def encontrar_mach_critico(df):
    """
    Encontra o valor de Mach crítico com base nos valores calculados.

    Parâmetros:
    df (pd.DataFrame): Um DataFrame com os valores de Mach crítico calculados.

    Retorno:
    pd.DataFrame: Um DataFrame contendo o valor de Mach crítico encontrado.
    """
    df_final_analitico = df[(df['mcr'] > 0.585656) & (df['mcr'] < 0.59)]
    return df_final_analitico[df_final_analitico['diff'] == df_final_analitico['diff'].min()]


#Plot o grafico do mach critico
def plotar_grafico_mach_critico(df):
    """
    Plota um gráfico do Mach crítico em função de x, destacando a interseção.

    Parâmetros:
    df (pd.DataFrame): Um DataFrame com os valores de Mach crítico.

    Retorno:
    None
    """
    mcr_plot = np.arange(0.01, 1, 0.001)
    cp_plot_list = []
    cp_2_plot_list = []

    df_plot = pd.DataFrame(columns=['mcr', 'cp', 'cp_2'])

    for i in mcr_plot:
        cp = cp_min / (np.sqrt(1 - i**2))
        cp_2 = (2 / (gama * ((i**2))) ) * (( ( (1 + relacao_gama * (i**2) ) / (1 + relacao_gama) ) ** (gama/(gama - 1)) ) - 1)
        cp_plot_list.append(cp)
        cp_2_plot_list.append(cp_2)

    df_plot['mcr'] = mcr_plot
    df_plot['cp'] = cp_plot_list
    df_plot['cp_2'] = cp_2_plot_list

    df_plot_eq7 = df_plot[df_plot['mcr'] >= 0.5]
    df_plot_eq20 = df_plot[df_plot['mcr'] < 0.8]

    plt.figure(figsize=(15, 6))
    plt.plot(df_plot_eq20['mcr'], -df_plot_eq20['cp'], label='cp', color='blue')
    plt.plot(df_plot_eq7['mcr'], -df_plot_eq7['cp_2'], label='cp_2', color='red')

    plt.scatter(0.585, 1.389497, c='green', label='Interseção')

    plt.xlabel('mcr')
    plt.ylabel('-cp')
    plt.legend()
    plt.annotate(f'Interseção: {0.585}, {1.389497}', 
                 (0.585, 1.389497),
                 textcoords="offset points",
                 xytext=(5,40),
                 ha='center')
    x_intersec = 0.585
    y_intersec = 1.389497

    plt.plot([x_intersec, x_intersec], [0, y_intersec], linestyle='--', color='gray')
    plt.show()


def calcular_cp_infinito(data, mach_inf):
    """
    Calcula o Cp para diferentes valores de Mach e armazena os resultados no DataFrame.

    Parâmetros:
    data (pd.DataFrame): O DataFrame com os dados do perfil aerodinâmico.
    mach_inf (list): Uma lista de valores de Mach para os quais o Cp será calculado.

    Retorno:
    None
    """
    for i in mach_inf:
        data[f'Cp_inf_{i}'] = data['Cp'] / np.sqrt(1 - i**2)

def plotar_grafico_mach_infinito(data, mach_inf, m_critico):
    """
    Plota gráficos de Cp infinito em função de x para diferentes valores de Mach, destacando a interseção.

    Parâmetros:
    data (pd.DataFrame): O DataFrame com os dados do perfil aerodinâmico.
    mach_inf (list): Uma lista de valores de Mach para os quais os gráficos serão gerados.
    m_critico (float): O valor de Mach crítico.

    Retorno:
    None
    """
    for i in mach_inf:
        data[f'-Cp_inf_{i}'] = -data[f'Cp_inf_{i}']

        positivos = data[data['y'] > 0]
        negativos = data[data['y'] < 0]

        plt.figure(figsize=(15, 8))
        plt.plot(positivos['x'], positivos[f'-Cp_inf_{i}'], linestyle='-', color='blue', label='C_p,u')
        plt.plot(negativos['x'], negativos[f'-Cp_inf_{i}'], linestyle='-', color='red', label='C_p,l')
        plt.axhline(m_critico, color='orange', linestyle='-.', label=f'-Cp = {m_critico}')
        plt.title(f'Gráfico de Linha Contínua de Cp com mach igual a: {i} em Função de x')
        plt.xlabel('x/c')
        plt.ylabel('-Cp')
        plt.grid(True)
        plt.legend()
        plt.show()

# Constante de conversão graus => radianos
D2R = np.pi / 180

# Load NACA 2413 data
def carregar_dados_perfil_ar(filename):
    """
    Carrega dados de perfil aerodinâmico de um arquivo e retorna coordenadas x, y e Cp.

    Parâmetros:
    filename (str): O nome do arquivo de dados.

    Retorno:
    x (numpy.array): Coordenadas x do perfil aerodinâmico.
    y (numpy.array): Coordenadas y do perfil aerodinâmico.
    cp (numpy.array): Valores de Cp (pressão) do perfil aerodinâmico.
    """
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    cp = data[:, 2]
    return x, y, cp

# Divide the airfoil into upper and lower surfaces
def dividir_perfil_asa(x, y):
    """
    Divide o perfil aerodinâmico em superfícies superior e inferior.

    Parâmetros:
    x (numpy.array): Coordenadas x do perfil aerodinâmico.
    y (numpy.array): Coordenadas y do perfil aerodinâmico.

    Retorno:
    x1 (numpy.array): Coordenadas x da superfície superior.
    y1 (numpy.array): Coordenadas y da superfície superior.
    x2 (numpy.array): Coordenadas x da superfície inferior.
    y2 (numpy.array): Coordenadas y da superfície inferior.
    """
    x1 = x[:len(x) // 2]
    x2 = x[len(x) // 2:]
    y1 = y[:len(y) // 2]
    y2 = y[len(y) // 2:]
    return x1, x2, y1, y2

# Calculate the derivative of y with respect to x
def calcular_derivada(y):

    """
    Faz o calculo da derivada

    """
    dydx = np.diff(y)
    return dydx

# Calculate the derivative of t and zc with respect to x
def calcular_derivadas_t_zc(y1, y2):
    """
    Calcula as derivadas de t e zc em relação a x.

    Parâmetros:
    y1 (numpy.array): Coordenadas da superfície superior.
    y2 (numpy.array): Coordenadas da superfície inferior.

    Retorno:
    t (numpy.array): Espessura do perfil aerodinâmico.
    zc (numpy.array): Centro de espessura do perfil aerodinâmico.
    dtdx (numpy.array): Derivada de t em relação a x.
    dzcdx (numpy.array): Derivada de zc em relação a x.
    """
    t = np.abs(y1 - y2)
    zc = t / 2
    dtdx = calcular_derivada(t)
    dzcdx = calcular_derivada(zc)
    return t, zc, dtdx, dzcdx

# Calculate the pressure coefficient (cp) based on Mach number for upper and lower surfaces
def calcular_cp(Mach, alfa, dydx_extr, dydx_intr):
    """
    Calcula o coeficiente de pressão (cp) com base no número de Mach para as superfícies superior e inferior.

    Parâmetros:
    Mach (numpy.array): Valores de número de Mach.
    alfa (float): Ângulo de ataque.
    dydx_extr (numpy.array): Derivada da coordenada y para a superfície superior.
    dydx_intr (numpy.array): Derivada da coordenada y para a superfície inferior.

    Retorno:
    cpu (numpy.array): Matriz de coeficiente de pressão (cp) para a superfície superior.
    cpl (numpy.array): Matriz de coeficiente de pressão (cp) para a superfície inferior.
    """
    cpu = np.zeros((len(Mach), len(dydx_extr)))
    cpl = np.zeros((len(Mach), len(dydx_intr)))
    
    for i in range(len(Mach)):
        cpu[i, 1:] = 2 * (dydx_extr[1:] - alfa) / np.sqrt(Mach[i] ** 2 - 1)
        cpl[i, 1:] = 2 * (dydx_intr[1:] - alfa) / np.sqrt(Mach[i] ** 2 - 1)
    
    return cpu, cpl

# Calculate lift coefficient (cl) based on Mach number
def calcular_cl(Mach, alfa):
    
    """
    Calcula o coeficiente de sustentação (cl) com base no número de Mach.

    Parâmetros:
    Mach (numpy.array): Valores de número de Mach.
    alfa (float): Ângulo de ataque.

    Retorno:
    cl (numpy.array): Coeficiente de sustentação (cl).
    """

    cl = (4* alfa) / (np.sqrt(Mach ** 2 - 1))
    return cl

# Calculate drag coefficient (cd) based on Mach number
def calcular_cd(Mach, alfa, dzcdx, dtdx, dxc):

    """
    Calcula o coeficiente de arrasto (cd) com base no número de Mach.

    Parâmetros:
    Mach (numpy.array): Valores de número de Mach.
    alfa (float): Ângulo de ataque.
    dzcdx (numpy.array): Derivada de zc em relação a x.
    dtdx (numpy.array): Derivada de t em relação a x.
    dxc (float): Passo de integração.

    Retorno:
    cd (numpy.array): Coeficiente de arrasto (cd).
    """

    cd = (1 / np.sqrt(Mach ** 2 - 1)) * (4 * alfa ** 2 + 4 * (np.sum(dzcdx) ** 2) * dxc + (np.sum(dtdx) ** 2) * dxc)
    return cd

# Plot cl vs Mach
def plot_cl_vs_mach(Mach, cl):
    """
    Plota o coeficiente de sustentação (cl) em função do número de Mach.

    Parâmetros:
    Mach (numpy.array): Valores de número de Mach.
    cl (numpy.array): Coeficiente de sustentação (cl).

    Retorno:
    None
    """


    plt.figure()
    plt.plot(Mach, cl, color='blue', linewidth=1.3)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(f'Gráfico do número de Cl em função do número de Mach')    
    plt.xlabel('Mach')
    plt.ylabel('cl')

# Plot cd vs Mach
def plot_cd_vs_mach(Mach, cd):

    """
    Plota o coeficiente de arrasto (cd) em função do número de Mach.

    Parâmetros:
    Mach (numpy.array): Valores de número de Mach.
    cd (numpy.array): Coeficiente de arrasto (cd).

    Retorno:
    None
    """


    plt.figure()
    plt.plot(Mach, cd, color='blue', linewidth=1.3)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(f'Gráfico do número de Cd em função do número de Mach')
    plt.xlabel('Mach')
    plt.ylabel('cd')



caminho_arquivo = "naca2.dat"
data = ler_arquivo(caminho_arquivo)
plotar_grafico(data)
    
mach_inf = [0.5, 0.6, 0.7, 0.8, 0.9]
calcular_cp_infinito(data, mach_inf)
    
m_critico = 1.389497
plotar_grafico_mach_infinito(data, mach_inf, m_critico)


# Suprimir avisos de SettingWithCopyWarning
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Especifique o caminho para o seu arquivo .dat
caminho_arquivo = "naca2.dat"

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Leitura do arquivo .dat usando o Pandas, considerando espaços em branco como separadores e dando nomes às colunas
data = pd.read_csv(caminho_arquivo, delim_whitespace=True, names=["x", "y", "Cp"])

alpha = np.radians(3)
data_cp = pd.read_csv('cp.csv')
data_cp.drop('-Cp', axis=1, inplace=True)

# Separando valores de y positivos e negativos
data_cp_y_positivo = data_cp[data_cp['y'] >= 0].rename(columns={
    'y': 'y_positivo',
    'x': 'x_positivo',
    'Cp': 'Cp_u',
    'Cp_inf_0.5': 'Cp_inf_0.5_u',
    'Cp_inf_0.6': 'Cp_inf_0.6_u',
    'Cp_inf_0.7': 'Cp_inf_0.7_u',
    'Cp_inf_0.8': 'Cp_inf_0.8_u',
    'Cp_inf_0.9': 'Cp_inf_0.9_u',
})
data_cp_y_negativo = data_cp[data_cp['y'] <= 0].rename(columns={
    'y': 'y_negativo',
    'x': 'x_negativo',
    'Cp': 'Cp_l',
    'Cp_inf_0.5': 'Cp_inf_0.5_l',
    'Cp_inf_0.6': 'Cp_inf_0.6_l',
    'Cp_inf_0.7': 'Cp_inf_0.7_l',
    'Cp_inf_0.8': 'Cp_inf_0.8_l',
    'Cp_inf_0.9': 'Cp_inf_0.9_l',
})

data_cp_y_negativo.reset_index(inplace=True, drop=True)
merge = pd.concat([data_cp_y_positivo, data_cp_y_negativo], axis=1)
merge.dropna(inplace=True)

# Cálculos
merge['x_positivo_lag'] = merge['x_positivo'].shift(periods=-1)
merge['y_positivo_lag'] = merge['y_positivo'].shift(periods=-1)
merge['x_negativo_lag'] = merge['x_negativo'].shift(periods=-1)
merge['y_negativo_lag'] = merge['y_negativo'].shift(periods=-1)
merge['dy_u/dx'] = (merge['y_positivo_lag'] - merge['y_positivo']) / (merge['x_positivo_lag'] - merge['x_positivo'])
merge['dy_l/dx'] = (merge['y_negativo_lag'] - merge['y_negativo']) / (merge['x_negativo_lag'] - merge['x_negativo'])

# Cálculos cn
cn_5 = ((merge['Cp_inf_0.5_l'] - merge['Cp_inf_0.5_u']) * (merge['x_positivo'] - merge['x_positivo_lag'])).sum()
cn_6 = ((merge['Cp_inf_0.6_l'] - merge['Cp_inf_0.6_u']) * (merge['x_positivo'] - merge['x_positivo_lag'])).sum()
cn_7 = ((merge['Cp_inf_0.7_l'] - merge['Cp_inf_0.7_u']) * (merge['x_positivo'] - merge['x_positivo_lag'])).sum()
cn_8 = ((merge['Cp_inf_0.8_l'] - merge['Cp_inf_0.8_u']) * (merge['x_positivo'] - merge['x_positivo_lag'])).sum()
cn_9 = ((merge['Cp_inf_0.9_l'] - merge['Cp_inf_0.9_u']) * (merge['x_positivo'] - merge['x_positivo_lag'])).sum()

# Cálculos ca
merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.5'] = merge['Cp_inf_0.5_u'] * merge['dy_u/dx'] - merge['Cp_inf_0.5_l'] * merge['dy_l/dx']
merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.6'] = merge['Cp_inf_0.6_u'] * merge['dy_u/dx'] - merge['Cp_inf_0.6_l'] * merge['dy_l/dx']
merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.7'] = merge['Cp_inf_0.7_u'] * merge['dy_u/dx'] - merge['Cp_inf_0.7_l'] * merge['dy_l/dx']
merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.8'] = merge['Cp_inf_0.8_u'] * merge['dy_u/dx'] - merge['Cp_inf_0.8_l'] * merge['dy_l/dx']
merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.9'] = merge['Cp_inf_0.9_u'] * merge['dy_u/dx'] - merge['Cp_inf_0.9_l'] * merge['dy_l/dx']

merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.5_lag'] = merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.5'].shift(periods=-1)
merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.6_lag'] = merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.6'].shift(periods=-1)
merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.7_lag'] = merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.7'].shift(periods=-1)
merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.8_lag'] = merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.8'].shift(periods=-1)
merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.9_lag'] = merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.9'].shift(periods=-1)
ca_5 = ((merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.5_lag'] + merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.5']) / 2 * (merge['x_positivo'] - merge['x_positivo_lag'] ) / 2).sum()
ca_6 = ((merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.6_lag'] + merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.6']) / 2 * (merge['x_positivo'] - merge['x_positivo_lag'] ) / 2).sum()
ca_7 = ((merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.7_lag'] + merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.7']) / 2 * (merge['x_positivo'] - merge['x_positivo_lag'] ) / 2).sum()
ca_8 = ((merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.8_lag'] + merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.8']) / 2 * (merge['x_positivo'] - merge['x_positivo_lag'] ) / 2).sum()
ca_9 = ((merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.9_lag'] + merge['Cpu(dyu/dx)-Cpl(dyl/dx)_0.9']) / 2 * (merge['x_positivo'] - merge['x_positivo_lag'] ) / 2).sum()

cl_5 = (cn_5 * np.cos(alpha)) - (ca_5 * np.sin(alpha))
cl_6 = (cn_6 * np.cos(alpha)) - (ca_6 * np.sin(alpha))
cl_7 = (cn_7 * np.cos(alpha)) - (ca_7 * np.sin(alpha))
cl_8 = (cn_8 * np.cos(alpha)) - (ca_8 * np.sin(alpha))
cl_9 = (cn_9 * np.cos(alpha)) - (ca_9 * np.sin(alpha))

cd_5 = (cn_5 * np.sin(alpha)) + (ca_5 * np.cos(alpha))
cd_6 = (cn_6 * np.sin(alpha)) + (ca_6 * np.cos(alpha))
cd_7 = (cn_7 * np.sin(alpha)) + (ca_7 * np.cos(alpha))
cd_8 = (cn_8 * np.sin(alpha)) + (ca_8 * np.cos(alpha))
cd_9 = (cn_9 * np.sin(alpha)) + (ca_9 * np.cos(alpha))

df = pd.DataFrame(columns=['mach', 'cl', 'cd'])
df.loc[0] = [0.5, cl_5, cd_5]
df.loc[1] = [0.6, cl_6, cd_6]
df.loc[2] = [0.7, cl_7, cd_7]
df.loc[3] = [0.8, cl_8, cd_8]
df.loc[4] = [0.9, cl_9, cd_9]

plt.figure(figsize=(12, 6))
plt.plot(df['mach'], df['cl'])
plt.title(f'Número de Cl x número de Mach')
plt.xlabel('Mach')
plt.ylabel('Cl')
plt.grid(True)
plt.legend()
plt.show()

# Continuação do código, não foi alterada
data2 = data.copy()
data2['x_lag'] = data2['x'].shift(periods=-1)
data2['y_lag'] = data2['y'].shift(periods=-1)
data2['dy/dx'] = (data2['y_lag'] - data2['y']) / (data2['x_lag'] - data2['x'])
data2['theta'] = data2['dy/dx']
mach_list = [1.1, 1.2, 1.3, 1.4, 1.5]

for i in mach_list:
    data2[f'cp_inf_mach_{i}'] = (2 * data2['theta']) / np.sqrt((i ** 2 - 1))
data2 = data2[data2['y'] != -0.00064]

for i in mach_list:
    data2[f'-cp_inf_mach_{i}'] = -data2[f'cp_inf_mach_{i}']

    # Crie duas séries para valores de y positivos e negativos
    positivos = data2[data2['y'] > 0]
    negativos = data2[data2['y'] < 0]

    # Defina os limites do eixo y para -2 e 2

    # Crie um gráfico de linha contínua com cores diferentes para valores de y positivos e negativos
    plt.figure(figsize=(12, 6))
    plt.ylim(-2, 2)
    plt.plot(positivos['x'], positivos[f'-cp_inf_mach_{i}'], linestyle='-', color='blue', label='C_p,u')
    plt.plot(negativos['x'], negativos[f'cp_inf_mach_{i}'], linestyle='-', color='red', label='C_p,l')
    plt.title(f'Gráfico de Linha Contínua de Cp com mach igual a: {i} em Função de x')
    plt.xlabel('x')
    plt.ylabel('-Cp')
    plt.grid(True)
    plt.legend()
    plt.show()




# Load airfoil data
x, y, cp = carregar_dados_perfil_ar('data.dat')

# Divide airfoil into upper and lower surfaces
x1, x2, y1, y2 = dividir_perfil_asa(x, y)

alfa = 3 * D2R  # ângulo de ataque
    
# Calculate derivatives and t, zc values
t, zc, dtdx, dzcdx = calcular_derivadas_t_zc(y1, y2)

    # Calculate cp based on Mach for upper and lower surfaces
Mach = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
dydx_extr = calcular_derivada(y1)
dydx_intr = calcular_derivada(y2)
cpu, cpl = calcular_cp(Mach, alfa, dydx_extr, dydx_intr)
    
    # Calculate cl and cd
cl = calcular_cl(Mach, alfa)
dxc = np.diff(x)[-1]
cd = calcular_cd(Mach, alfa, dzcdx, dtdx, dxc)
    
    # Plot cl vs Mach
plot_cl_vs_mach(Mach, cl)
    
    # Plot cd vs Mach
plot_cd_vs_mach(Mach, cd)
    
plt.show()


