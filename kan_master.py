import os
import sys
import pickle
import re
import warnings
import numpy as np
import scipy.io
import sympy
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from collections import Counter

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.covariance import GraphicalLasso
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from pygsp import graphs
from kan import KAN

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")
np.random.seed(42)

plt.rcParams.update({
    "font.family": "serif", "font.size": 12, "figure.dpi": 300, 
    "axes.grid": True, "grid.alpha": 0.3
})

DIR_CHK = r"./checkpoints_tesis"
DIR_FIGS = r"./figuras_tesis"
os.makedirs(DIR_CHK, exist_ok=True)
os.makedirs(DIR_FIGS, exist_ok=True)

archivo_dataset_crudo = os.path.join(DIR_CHK, "dataset_crudo_master.pkl")
archivo_checkpoint_paso2 = os.path.join(DIR_CHK, "checkpoint_paso2.pkl")
archivo_dataset_pt = os.path.join(DIR_CHK, "dataset_tensores_fisica.pt")
archivo_modelo_binario = os.path.join(DIR_CHK, "modelo_kan_binario.pt")
archivo_historial = os.path.join(DIR_CHK, "historial_optimizacion_binario.pkl")
archivo_dataset_mc = os.path.join(DIR_CHK, "dataset_tensores_mc_fisico.pt")
archivo_modelo_multiclase = os.path.join(DIR_CHK, "modelo_kan_multiclase.pt")
ruta_ecuacion = os.path.join(DIR_CHK, "ecuaciones_frontera.txt")
archivo_stats_simbolicas = os.path.join(DIR_CHK, "stats_simbolicas.pkl")


def robust_split(X, Y):
    """Partición segura contra colapsos por sets de datos extremadamente pequeños."""
    if len(X) < 2:
        return X, X, Y, Y
    try:
        return train_test_split(X, Y, test_size=0.20, random_state=42, stratify=Y)
    except ValueError:
        try:
            return train_test_split(X, Y, test_size=0.20, random_state=42)
        except ValueError:
            return X, X, Y, Y

## Step 1: Data Loading
nombres_clases_dinamicas = ["Normal"]
mapeo_carpetas = {}
idx_clase_actual = 1  

if os.path.exists(archivo_dataset_crudo):
    with open(archivo_dataset_crudo, "rb") as f:
        datos = pickle.load(f)
    dataset_crudo = datos['dataset_crudo']
    W_fisica = datos['W_fisica']
    mascara_gen = datos['mascara_gen']
    mascara_load = datos['mascara_load']
    indices_buses_validos = datos['indices_buses_validos']
    
    if dataset_crudo and any(e['etiqueta'] == 0 for e in dataset_crudo):
        for e in dataset_crudo: e['etiqueta'] += 1
        with open(archivo_dataset_crudo, "wb") as f:
            pickle.dump({'dataset_crudo': dataset_crudo, 'W_fisica': W_fisica, 'mascara_gen': mascara_gen, 'mascara_load': mascara_load, 'indices_buses_validos': indices_buses_validos}, f)
            
    max_etiq = max([e['etiqueta'] for e in dataset_crudo]) if dataset_crudo else 0
    for i in range(1, max_etiq + 1):
        if i == 1: nombres_clases_dinamicas.append("Generator Trip")
        elif i == 2: nombres_clases_dinamicas.append("Line Trip")
        elif i == 3: nombres_clases_dinamicas.append("Load Trip")
        else: nombres_clases_dinamicas.append(f"Event Class {i}")
    num_clases_totales = len(nombres_clases_dinamicas)
else:
    frecuencia_base, limite_desviacion = 60.0, 2.0 
    DIR_DATOS = r"C:\Users\PowerSystemLab\Dropbox\Tesis_MSc_Manuel\work\Simulaciones Tesis\add_rew\ME_SY\eventos_mexico"
    
    mat_topo = scipy.io.loadmat(os.path.join(DIR_DATOS, "90bus_wrew.mat"), squeeze_me=True, struct_as_record=False)
    matriz_bus_global = mat_topo['bus']
    matriz_line_global = mat_topo['line']

    archivos_mat = [os.path.join(r, a) for r, d, files in os.walk(DIR_DATOS) for a in files if a.endswith(".mat") and "90bus_" not in a and "Steady" not in a]
    dataset_crudo = []
    indices_buses_validos = None

    for archivo in tqdm(archivos_mat, desc="Processing MATs"):
        ruta = archivo.lower().replace('\\', '/')
        carpeta_padre = os.path.basename(os.path.dirname(archivo)).lower()
        
        # Mapeo dinámico de CUALQUIER carpeta 
        if carpeta_padre not in mapeo_carpetas:
            mapeo_carpetas[carpeta_padre] = idx_clase_actual
            if 'gen' in carpeta_padre: nombres_clases_dinamicas.append("Generator Trip")
            elif 'line' in carpeta_padre: nombres_clases_dinamicas.append("Line Trip")
            elif 'load' in carpeta_padre: nombres_clases_dinamicas.append("Load Trip")
            else: nombres_clases_dinamicas.append(carpeta_padre.replace('_', ' ').title() + " Event")
            idx_clase_actual += 1
            
        etiqueta = mapeo_carpetas[carpeta_padre]

        try:
            mat = scipy.io.loadmat(archivo, squeeze_me=True, struct_as_record=False)
            bus_freq = mat['sstr'].bus_freq * frecuencia_base
            bus_v = np.abs(mat['sstr'].bus_v)
            
            if indices_buses_validos is None:
                indices_buses_validos = [i for i in range(bus_freq.shape[0]) if np.max(np.abs(bus_freq[i, :] - frecuencia_base)) < limite_desviacion]
            
            bf_f, bv_f = bus_freq[indices_buses_validos, :], bus_v[indices_buses_validos, :]
            dataset_crudo.append({
                'nombre': os.path.basename(archivo), 'etiqueta': etiqueta, 't': mat['sstr'].t,
                'delta_f': bf_f + np.random.normal(0, 0.0005, bf_f.shape) - frecuencia_base,
                'delta_v': bv_f + np.random.normal(0, 0.0001, bv_f.shape) - 1.0 
            })
        except Exception as e:
            print(f"\n[Error] Failed to process {os.path.basename(archivo)}: {e}")
            continue
            
    if not dataset_crudo or indices_buses_validos is None:
        raise ValueError("Critical Error: No .mat files could be processed.")
        
    num_clases_totales = len(nombres_clases_dinamicas)

    N_buses = len(indices_buses_validos)
    W_fisica, mascara_gen, mascara_load = np.zeros((N_buses, N_buses)), np.zeros(N_buses), np.zeros(N_buses)

    matriz_line_global = mat_topo['line'] if isinstance(mat_topo, dict) else mat_topo.line
    matriz_bus_global = mat_topo['bus'] if isinstance(mat_topo, dict) else mat_topo.bus

    for fila in matriz_line_global:
        o, d = int(fila[0])-1, int(fila[1])-1
        if o in indices_buses_validos and d in indices_buses_validos:
            idx_o, idx_d = indices_buses_validos.index(o), indices_buses_validos.index(d)
            admitancia = 1.0 / np.sqrt(fila[2]**2 + fila[3]**2)
            W_fisica[idx_o, idx_d] = W_fisica[idx_d, idx_o] = admitancia

    for fila in matriz_bus_global:
        id_bus = int(fila[0])-1
        if id_bus in indices_buses_validos:
            idx = indices_buses_validos.index(id_bus)
            if int(fila[9]) in [1, 2] or id_bus < 46: mascara_gen[idx] = 1.0
            else: mascara_load[idx] = 1.0
                
    with open(archivo_dataset_crudo, "wb") as f:
        pickle.dump({'dataset_crudo': dataset_crudo, 'W_fisica': W_fisica, 'mascara_gen': mascara_gen,
                     'mascara_load': mascara_load, 'indices_buses_validos': indices_buses_validos}, f)

colores_dinamicos = [plt.cm.tab10(i % 10) for i in range(num_clases_totales)]

## Step 2: Topology inference
if os.path.exists(archivo_checkpoint_paso2):
    with open(archivo_checkpoint_paso2, "rb") as f:
        datos_paso2 = pickle.load(f)
        
    W_inferida = datos_paso2['W_inferida']
    W_final = datos_paso2['W_final']
    historial_alphas = datos_paso2['historial_alphas']
    historial_aristas = datos_paso2['historial_aristas']
    alpha_optimo = datos_paso2['alpha_optimo']
    aristas_optimas = datos_paso2['aristas_optimas']
    objetivo_aristas = datos_paso2['objetivo_aristas']
    G = graphs.Graph(W_final)

else:
    datos_estandarizados = StandardScaler().fit_transform(dataset_crudo[0]['delta_f'].T)
    rango_alphas = [0.8, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
    objetivo_aristas = (datos_estandarizados.shape[1] * 1.2, datos_estandarizados.shape[1] * 6)
    
    W_inferida = np.zeros((datos_estandarizados.shape[1], datos_estandarizados.shape[1]))
    historial_alphas, historial_aristas = [], []
    alpha_optimo, aristas_optimas = None, None

    for alpha in rango_alphas:
        try:
            modelo = GraphicalLasso(alpha=alpha, max_iter=1000, tol=1e-3, assume_centered=True)
            modelo.fit(datos_estandarizados)
            
            W_temp = np.abs(modelo.precision_)
            np.fill_diagonal(W_temp, 0)
            if np.max(W_temp) > 0:
                W_temp = W_temp / np.max(W_temp)
                W_temp[W_temp < 0.05] = 0 
                
            G_temp = graphs.Graph(W_temp)
            historial_alphas.append(alpha)
            historial_aristas.append(G_temp.Ne)
            
            if objetivo_aristas[0] <= G_temp.Ne <= objetivo_aristas[1] and alpha_optimo is None:
                W_inferida = W_temp
                alpha_optimo = alpha
                aristas_optimas = G_temp.Ne
        except Exception:
            continue

    if alpha_optimo is None and historial_alphas: W_inferida = W_temp

    W_fisica_norm = W_fisica / np.max(W_fisica) if np.max(W_fisica) > 0 else W_fisica
    W_final = (0.5 * W_fisica_norm) + (0.5 * W_inferida)
    W_final[W_final < 0.05] = 0
    np.fill_diagonal(W_final, 0)
    
    G = graphs.Graph(W_final)
    
    with open(archivo_checkpoint_paso2, "wb") as f:
        pickle.dump({
            'W_inferida': W_inferida, 'W_final': W_final, 'historial_alphas': historial_alphas,
            'historial_aristas': historial_aristas, 'alpha_optimo': alpha_optimo,
            'aristas_optimas': aristas_optimas, 'objetivo_aristas': objetivo_aristas
        }, f)

G.compute_fourier_basis()

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12, "axes.linewidth": 1.2,
    "legend.fontsize": 10, "legend.framealpha": 1.0, "legend.edgecolor": "black",
    "xtick.labelsize": 10, "ytick.labelsize": 10, "xtick.direction": "in", "ytick.direction": "in",
    "grid.alpha": 0.6, "grid.linestyle": "--"
})

# Plot 2A
fig_opt, ax_opt = plt.subplots(figsize=(8, 5))
ax_opt.plot(historial_alphas, historial_aristas, marker='o', markerfacecolor='none', markeredgecolor='blue', color='blue', linewidth=1.5, label="Inference")
ax_opt.axhspan(objetivo_aristas[0], objetivo_aristas[1], color='gray', alpha=0.2, label="Physical Target Range")

if alpha_optimo:
    ax_opt.scatter([alpha_optimo], [aristas_optimas], color='red', s=150, marker='*', edgecolors='black', zorder=5, label="Optimum")
    idx_texto = len(historial_alphas) // 2
    if len(historial_alphas) > 0:
        ax_opt.text(historial_alphas[idx_texto], objetivo_aristas[1] * 1.1, 
                    f'Heuristic target: {int(objetivo_aristas[0])} - {int(objetivo_aristas[1])} edges', 
                    color='black', fontweight='bold', fontsize=10)

ax_opt.set(xscale='log', xlabel='Alpha (Regularization)', ylabel='Inferred Edges', title='Graphical Lasso Optimization')
ax_opt.invert_xaxis()
ax_opt.grid(True, which='both', linestyle='--', linewidth=0.5)
ax_opt.legend()
fig_opt.savefig(os.path.join(DIR_FIGS, "Fig2_A_Optimization.pdf"), bbox_inches='tight')

# Plot 2B 
W_fisica_norm = W_fisica / np.max(W_fisica) if np.max(W_fisica) > 0 else W_fisica
fig_mat, axes_mat = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(W_fisica_norm, cmap="jet", mask=(W_fisica_norm == 0), ax=axes_mat[0]).set_title("Physical Topology")
sns.heatmap(W_inferida, cmap="jet", mask=(W_inferida == 0), ax=axes_mat[1]).set_title("Inferred Topology (Graphical Lasso)")
sns.heatmap(W_final, cmap="jet", mask=(W_final == 0), ax=axes_mat[2]).set_title("Hybrid Topology")

for ax in axes_mat:
    for _, spine in ax.spines.items():
        spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(1.2)

fig_mat.savefig(os.path.join(DIR_FIGS, "Fig2_B_Matrices.pdf"), bbox_inches='tight')

# Plot 2C
fig_spec, ax_spec = plt.subplots(figsize=(8, 5))
ax_spec.plot(range(len(G.e)), G.e, marker='s', markerfacecolor='none', markeredgecolor='black', color='black', linewidth=1.5)
ax_spec.axvline(x=1, color='red', linestyle='--', linewidth=1.5, label="Fiedler Value")
ax_spec.set(xlabel='Frequency (k)', ylabel='Eigenvalue', title='GFT Spectrum')
ax_spec.grid(True, linestyle='--', linewidth=0.5)
ax_spec.legend()
fig_spec.savefig(os.path.join(DIR_FIGS, "Fig2_C_Spectrum.pdf"), bbox_inches='tight')

def plot_graph_opcion3(ax, W, title, color_data='#1f77b4', vector_orden=None):
    g_nx = nx.from_numpy_array(W)
    if len(g_nx.nodes) == 0: return
    componentes = list(nx.connected_components(g_nx))
    if not componentes: return
    componente_principal = max(componentes, key=len)
    g_sub = g_nx.subgraph(componente_principal)
    
    if vector_orden is not None and len(vector_orden) >= len(g_nx.nodes):
        nodos_ordenados = sorted(list(g_sub.nodes()), key=lambda n: vector_orden[n])
        pos_base = nx.circular_layout(g_sub)
        pos = {nodo: pos_base[nodo_base] for nodo, nodo_base in zip(nodos_ordenados, g_sub.nodes())}
    else:
        pos = nx.circular_layout(g_sub)
    
    if isinstance(color_data, np.ndarray) and len(color_data) >= len(g_nx.nodes):
        colores = [color_data[nodo] for nodo in g_sub.nodes()]
        cmap = plt.cm.jet
    else:
        colores = color_data; cmap = None

    edges, weights = zip(*nx.get_edge_attributes(g_sub, 'weight').items()) if g_sub.edges else ([], [])
    widths = (np.array(weights) / np.max(weights)) * 2 if len(weights) > 0 else 0
    
    nx.draw(g_sub, pos, ax=ax, node_color=colores, cmap=cmap, node_size=300, 
            edgelist=edges, width=widths, edge_color='gray', alpha=0.8,
            with_labels=True, font_size=8, font_color='black', font_family='serif',
            edgecolors='black', linewidths=1.2)
    ax.set_title(title, fontweight='bold')

fig_topo, axes_topo = plt.subplots(1, 2, figsize=(14, 6))
plot_graph_opcion3(axes_topo[0], W_fisica_norm, "Physical Matrix", vector_orden=G.U[:, 1] if G.U.shape[1] > 1 else None)
plot_graph_opcion3(axes_topo[1], W_final, "Fiedler Vector (Global Oscillations)", color_data=G.U[:, 1] if G.U.shape[1] > 1 else 'blue', vector_orden=G.U[:, 1] if G.U.shape[1] > 1 else None)
plt.tight_layout()
fig_topo.savefig(os.path.join(DIR_FIGS, "Fig2_D_Modes.pdf"), bbox_inches='tight')

## Step 3: Spectral Data
if os.path.exists(archivo_dataset_pt):
    dataset = torch.load(archivo_dataset_pt, weights_only=False)
    X_train_np = dataset['train_input'].numpy()
    Y_train_np = dataset['train_label'].numpy()
else:
    X_datos, Y_etiquetas = [], []
    for evento in dataset_crudo:
        t, delta_f, delta_v = evento['t'], evento['delta_f'], evento['delta_v']
        
        idx_normal = np.argmin(np.abs(t - 0.5))
        espectro_v_norm = np.abs(G.gft(delta_v[:, idx_normal]))
        espectro_f_gen_norm = np.abs(G.gft(delta_f[:, idx_normal] * mascara_gen))
        espectro_f_load_norm = np.abs(G.gft(delta_f[:, idx_normal] * mascara_load))
        
        X_datos.append(np.concatenate([espectro_v_norm, espectro_f_gen_norm, espectro_f_load_norm]))
        Y_etiquetas.append([0.0])
        
        mascara_transitorio = t > 1.1
        espectro_v_post = np.max(np.abs(G.gft(delta_v[:, mascara_transitorio])), axis=1)
        espectro_f_gen_post = np.max(np.abs(G.gft(delta_f[:, mascara_transitorio] * mascara_gen[:, np.newaxis])), axis=1)
        espectro_f_load_post = np.max(np.abs(G.gft(delta_f[:, mascara_transitorio] * mascara_load[:, np.newaxis])), axis=1)
        
        X_datos.append(np.concatenate([espectro_v_post, espectro_f_gen_post, espectro_f_load_post]))
        Y_etiquetas.append([1.0])

    X_datos_np, Y_etiquetas_np = np.array(X_datos), np.array(Y_etiquetas)
    
    X_train, X_test, Y_train, Y_test = robust_split(X_datos_np, Y_etiquetas_np)
    
    dataset = {
        'train_input': torch.tensor(X_train, dtype=torch.float32),
        'train_label': torch.tensor(Y_train, dtype=torch.float32),
        'test_input': torch.tensor(X_test, dtype=torch.float32),
        'test_label': torch.tensor(Y_test, dtype=torch.float32),
    }
    torch.save(dataset, archivo_dataset_pt)
    X_train_np, Y_train_np = X_train, Y_train

N = G.N
evento = dataset_crudo[0]
idx_norm = np.argmin(np.abs(evento['t'] - 0.5))
mask_trans = evento['t'] > 1.1

spec_f_norm = np.abs(G.gft(evento['delta_f'][:, idx_norm]))
spec_f_trans = np.max(np.abs(G.gft(evento['delta_f'][:, mask_trans])), axis=1)

# Plot 3A
fig_spec, ax_spec = plt.subplots(figsize=(10, 5))
markerline1, stemlines1, baseline1 = ax_spec.stem(range(N), spec_f_trans, linefmt='r-', markerfmt='ro', basefmt=' ', label='Transient (Max)')
plt.setp(markerline1, markerfacecolor='none', markeredgecolor='red', markeredgewidth=1.2, markersize=6)
plt.setp(stemlines1, linewidth=1.5)

markerline2, stemlines2, baseline2 = ax_spec.stem(range(N), spec_f_norm, linefmt='b-', markerfmt='bo', basefmt=' ', label='Normal (t=0.5s)')
plt.setp(markerline2, markerfacecolor='none', markeredgecolor='blue', markeredgewidth=1.2, markersize=6)
plt.setp(stemlines2, linewidth=1.5, linestyle='--')

ax_spec.set(xlabel='Graph Frequency Index (k)', ylabel='Magnitude |GFT|', title='Frequency Spectral Contrast')
ax_spec.grid(True, which='both', linestyle='--', linewidth=0.5)
ax_spec.legend()
fig_spec.savefig(os.path.join(DIR_FIGS, "Fig3_A_Spectral_Contrast.pdf"), bbox_inches='tight')

# Plot 3B
transient_indices = np.where(Y_train_np == 1.0)[0]
idx_transitorio = transient_indices[0] if len(transient_indices) > 0 else 0
tensor_ejemplo = X_train_np[idx_transitorio]
y_max = np.max(tensor_ejemplo) if len(tensor_ejemplo) > 0 else 1

fig_tensor, ax_tensor = plt.subplots(figsize=(12, 4))
markerline3, stemlines3, baseline3 = ax_tensor.stem(range(len(tensor_ejemplo)), tensor_ejemplo, linefmt='#9467bd', markerfmt='o', basefmt=' ')
plt.setp(markerline3, markerfacecolor='none', markeredgecolor='#9467bd', markeredgewidth=1.2, markersize=5)
plt.setp(stemlines3, linewidth=1.5)

ax_tensor.axvline(x=N-0.5, color='k', linestyle='--', alpha=0.8, linewidth=1.2)
ax_tensor.axvline(x=2*N-0.5, color='k', linestyle='--', alpha=0.8, linewidth=1.2)

bbox_style = dict(facecolor='white', alpha=1.0, edgecolor='black', linewidth=1.0)
ax_tensor.text(N/2, y_max*0.9, r'$\Delta V$ Spectrum', ha='center', fontweight='bold', bbox=bbox_style)
ax_tensor.text(N + N/2, y_max*0.9, r'$\Delta f$ Spectrum (Gen)', ha='center', fontweight='bold', bbox=bbox_style)
ax_tensor.text(2*N + N/2, y_max*0.9, r'$\Delta f$ Spectrum (Load)', ha='center', fontweight='bold', bbox=bbox_style)

ax_tensor.set(xlabel=f'Tensor Index (Dimension = {3*N})', ylabel='Magnitude', title='Concatenated Tensor Anatomy')
ax_tensor.grid(True, linestyle='--', linewidth=0.5)
fig_tensor.savefig(os.path.join(DIR_FIGS, "Fig3_B_Tensor_Anatomy.pdf"), bbox_inches='tight')

# Plot 3C: PCA 
try:
    n_comp = min(2, X_train_np.shape[0], X_train_np.shape[1])
    if n_comp >= 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train_np)

        fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
        scatter = ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=Y_train_np.flatten(), cmap='coolwarm', edgecolors='k', alpha=0.9, s=60, linewidths=1.2)
        ax_pca.set(xlabel=f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                   ylabel=f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                   title='GFT Space PCA Projection')
        ax_pca.grid(True, linestyle='--', linewidth=0.5)

        cbar = plt.colorbar(scatter, ax=ax_pca, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Normal (0)', 'Transient (1)'])
        cbar.outline.set_linewidth(1.2)
        fig_pca.savefig(os.path.join(DIR_FIGS, "Fig3_C_PCA_Separability.pdf"), bbox_inches='tight')
except Exception as e:
    print(f"\n[Warning] Skipped PCA Plot due to insufficient data: {e}")

## Step 4: Binary KAN
dimension_entrada = G.N * 3
modelo_kan = KAN(width=[dimension_entrada, 3, 1], grid=5, k=3, seed=42)
modelo_kan_inicial = KAN(width=[dimension_entrada, 3, 1], grid=5, k=3, seed=42)

if os.path.exists(archivo_modelo_binario) and os.path.exists(archivo_historial):
    modelo_kan.load_state_dict(torch.load(archivo_modelo_binario, weights_only=True))
    with open(archivo_historial, "rb") as f:
        resultados_entrenamiento = pickle.load(f)
else:
    if len(dataset['train_input']) > 0:
        resultados_entrenamiento = modelo_kan.fit(dataset, opt="LBFGS", steps=20, log=10)
        torch.save(modelo_kan.state_dict(), archivo_modelo_binario)
        with open(archivo_historial, "wb") as f:
            pickle.dump(resultados_entrenamiento, f)

predicciones = modelo_kan(dataset['test_input']).detach()
y_true = dataset['test_label'].detach().numpy().flatten()
y_pred_cont = predicciones.numpy().flatten()
y_pred_bin = np.clip(torch.round(predicciones).numpy().flatten(), 0, 1)

fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))

cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
cm_sums = cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.divide(cm.astype('float'), cm_sums, out=np.zeros_like(cm, dtype=float), where=cm_sums!=0)

sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="jet", ax=axes4[0],
            xticklabels=["Normal", "Transient"], yticklabels=["Normal", "Transient"],
            linewidths=1, linecolor='black', vmin=0, vmax=1)
axes4[0].set_xlabel('KAN Prediction', fontweight='bold')
axes4[0].set_ylabel('True Condition', fontweight='bold')
axes4[0].set_title('Confusion Matrix', fontweight='bold')

for _, spine in axes4[0].spines.items():
    spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(1.2)

if len(np.unique(y_true)) > 1:
    fpr, tpr, _ = roc_curve(y_true, y_pred_cont)
    label_roc = f'ROC (AUC = {auc(fpr, tpr):.4f})'
else:
    fpr, tpr, label_roc = [0, 1], [0, 1], 'ROC (N/A - Insufficient Data)'
    
axes4[1].plot(fpr, tpr, color='red', lw=2, label=label_roc)
axes4[1].plot([0, 1], [0, 1], color='black', lw=1.5, linestyle='--')
axes4[1].set_xlabel('False Positive Rate', fontweight='bold')
axes4[1].set_ylabel('True Positive Rate', fontweight='bold')
axes4[1].set_title('ROC Curve', fontweight='bold')
axes4[1].legend(loc="lower right")
axes4[1].grid(True, linestyle='--', linewidth=0.5)

normal_preds = y_pred_cont[y_true == 0]
transient_preds = y_pred_cont[y_true == 1]

if len(normal_preds) > 1 and np.var(normal_preds) > 1e-6:
    sns.kdeplot(normal_preds, ax=axes4[2], fill=True, color="blue", alpha=0.3, linewidth=2, label="True: Normal")
elif len(normal_preds) > 0:
    axes4[2].axvline(normal_preds[0], color="blue", label="True: Normal (Low variance)")

if len(transient_preds) > 1 and np.var(transient_preds) > 1e-6:
    sns.kdeplot(transient_preds, ax=axes4[2], fill=True, color="red", alpha=0.3, linewidth=2, label="True: Transient")
elif len(transient_preds) > 0:
    axes4[2].axvline(transient_preds[0], color="red", label="True: Transient (Low variance)")

axes4[2].axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='Threshold (0.5)')
axes4[2].set_xlabel('Continuous Output', fontweight='bold')
axes4[2].set_ylabel('Density', fontweight='bold')
axes4[2].set_title('Certainty Distribution', fontweight='bold')
axes4[2].grid(True, linestyle='--', linewidth=0.5)
axes4[2].legend()

plt.tight_layout()
fig4.savefig(os.path.join(DIR_FIGS, "Fig4_A_KAN_Evaluation.pdf"), bbox_inches='tight')

# Plot 4B 
try:
    fig_opt, ax_opt = plt.subplots(1, 2, figsize=(12, 5))
    pasos = range(1, len(resultados_entrenamiento['train_loss']) + 1)

    ax_opt[0].plot(pasos, resultados_entrenamiento['train_loss'], 'b-', lw=2, label='Train Loss')
    ax_opt[0].plot(pasos, resultados_entrenamiento['test_loss'], 'r--', lw=2, label='Test Loss')
    ax_opt[0].set_xlabel('Iterations', fontweight='bold')
    ax_opt[0].set_ylabel('Mean Squared Error', fontweight='bold')
    ax_opt[0].set_title('L-BFGS Convergence', fontweight='bold')
    ax_opt[0].grid(True, linestyle='--', linewidth=0.5)
    ax_opt[0].legend()

    ax_opt[1].plot(pasos, resultados_entrenamiento['reg'], 'g-', lw=2)
    ax_opt[1].set_xlabel('Iterations', fontweight='bold')
    ax_opt[1].set_ylabel('Penalty (Sparsity)', fontweight='bold')
    ax_opt[1].set_title('Learned Parameters Evolution', fontweight='bold')
    ax_opt[1].grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    fig_opt.savefig(os.path.join(DIR_FIGS, "Fig4_B_Optimization_Dynamics.pdf"), bbox_inches='tight')
except Exception:
    pass

# Plot 4D
try:
    x_sweep = torch.linspace(-2, 2, 100).unsqueeze(1)
    act_inicial = modelo_kan_inicial.act_fun[0](x_sweep)[0][:, 0].detach().numpy()
    act_final = modelo_kan.act_fun[0](x_sweep)[0][:, 0].detach().numpy()

    fig_spline, ax_spline = plt.subplots(figsize=(6, 5))
    ax_spline.plot(x_sweep.numpy(), act_inicial, color='gray', linestyle='--', lw=2.5, label='Initial (Untrained)')
    ax_spline.plot(x_sweep.numpy(), act_final, color='blue', lw=2.5, label='Final (Learned)')

    ax_spline.set_xlabel('Input Value (x)', fontweight='bold')
    ax_spline.set_ylabel(r'Learned Function $\phi(x)$', fontweight='bold')
    ax_spline.set_title('KAN Spline Curve Deformation', fontweight='bold')
    ax_spline.grid(True, linestyle='--', linewidth=0.5)
    ax_spline.legend()
    fig_spline.savefig(os.path.join(DIR_FIGS, "Fig4_D_Spline_Evolution.pdf"), bbox_inches='tight')
except Exception:
    pass

## Step 5: Multiclass dataset 
if os.path.exists(archivo_dataset_mc):
    dataset_mc = torch.load(archivo_dataset_mc, weights_only=False)
    X_train_mc = dataset_mc['train_input'].numpy()
    Y_train_mc = dataset_mc['train_label'].numpy()
else:
    X_datos_mc, Y_etiquetas_mc = [], []
    for evento in dataset_crudo:
        t, delta_f, delta_v = evento['t'], evento['delta_f'], evento['delta_v']
        clase_original = evento['etiqueta'] 
        
        ventanas_norm = [(0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        carac_norm = []
        for t_ini, t_fin in ventanas_norm:
            mascara_n = (t > t_ini) & (t <= t_fin)
            carac_norm.extend([
                np.max(np.abs(G.gft(delta_v[:, mascara_n])), axis=1),
                np.max(np.abs(G.gft(delta_f[:, mascara_n] * mascara_gen[:, np.newaxis])), axis=1),
                np.max(np.abs(G.gft(delta_f[:, mascara_n] * mascara_load[:, np.newaxis])), axis=1)
            ])
        X_datos_mc.append(np.concatenate(carac_norm))
        
        y_norm = [0.0] * num_clases_totales
        y_norm[0] = 1.0
        Y_etiquetas_mc.append(y_norm) 
        
        ventanas_trans = [(1.1, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0)]
        carac_post = []
        for t_ini, t_fin in ventanas_trans:
            mascara_t = (t > t_ini) & (t <= t_fin)
            carac_post.extend([
                np.max(np.abs(G.gft(delta_v[:, mascara_t])), axis=1),
                np.max(np.abs(G.gft(delta_f[:, mascara_t] * mascara_gen[:, np.newaxis])), axis=1),
                np.max(np.abs(G.gft(delta_f[:, mascara_t] * mascara_load[:, np.newaxis])), axis=1)
            ])
        X_datos_mc.append(np.concatenate(carac_post))
        
        y_trans = [0.0] * num_clases_totales
        y_trans[clase_original] = 1.0 
        Y_etiquetas_mc.append(y_trans)

    X_np, Y_np = np.array(X_datos_mc), np.array(Y_etiquetas_mc)
    X_train, X_test, Y_train, Y_test = robust_split(X_np, Y_np)
    
    dataset_mc = {
        'train_input': torch.tensor(X_train, dtype=torch.float32),
        'train_label': torch.tensor(Y_train, dtype=torch.float32),
        'test_input': torch.tensor(X_test, dtype=torch.float32),
        'test_label': torch.tensor(Y_test, dtype=torch.float32),
    }
    torch.save(dataset_mc, archivo_dataset_mc)
    X_train_mc, Y_train_mc = X_train, Y_train

# Plot 5A 
line_indices = np.where(Y_train_mc[:, min(1, num_clases_totales - 1)] == 1.0)[0]
idx_linea = line_indices[0] if len(line_indices) > 0 else 0

if len(X_train_mc) > 0:
    tensor_linea = X_train_mc[idx_linea]
    matriz_evolucion = np.column_stack([tensor_linea[(w*3*N) : (w*3*N)+N] for w in range(5)])

    fig_cin, ax_cin = plt.subplots(figsize=(8, 6))
    sns.heatmap(matriz_evolucion, cmap="jet", ax=ax_cin, 
                xticklabels=["W1 (1-2s)", "W2 (2-5s)", "W3 (5-10s)", "W4 (10-15s)", "W5 (15-20s)"],
                cbar_kws={'label': r'Magnitude |GFT| of $\Delta V$'})

    ax_cin.set_xlabel('Time Windows', fontweight='bold')
    ax_cin.set_ylabel('Graph Frequency (k)', fontweight='bold')
    ax_cin.set_title('Kinematic Spectrum Evolution', fontweight='bold')

    for _, spine in ax_cin.spines.items():
        spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(1.2)
    fig_cin.savefig(os.path.join(DIR_FIGS, "Fig5_A_Kinematic_Evolution.pdf"), bbox_inches='tight')

# Plot 5B 
conteo_clases = np.sum(Y_train_mc, axis=0) + np.sum(dataset_mc['test_label'].numpy(), axis=0)

fig_bal_mc, ax_bal_mc = plt.subplots(figsize=(7, 5))
ax_bal_mc.grid(axis='y', linestyle='--', linewidth=0.5, zorder=0)

barras_mc = ax_bal_mc.bar(nombres_clases_dinamicas, conteo_clases, color=colores_dinamicos, 
                          edgecolor='black', linewidth=1.2, zorder=3)
ax_bal_mc.bar_label(barras_mc, padding=3, fontweight='bold', family='serif')

ax_bal_mc.set_ylabel('Total Samples', fontweight='bold')
ax_bal_mc.set_title('Final Multiclass Dataset Distribution', fontweight='bold')
plt.xticks(rotation=45, ha='right')

for _, spine in ax_bal_mc.spines.items():
    spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(1.2)

fig_bal_mc.savefig(os.path.join(DIR_FIGS, "Fig5_B_Multiclass_Balance.pdf"), bbox_inches='tight')

## Step 6: Multiclass KAN
dataset_mc = torch.load(archivo_dataset_mc, weights_only=False)
X_test_tensor = dataset_mc['test_input']
Y_test_tensor = dataset_mc['test_label']

dimension_entrada_mc = X_test_tensor.shape[1] if len(X_test_tensor) > 0 else G.N * 15
modelo_kan_mc = KAN(width=[dimension_entrada_mc, 8, num_clases_totales], grid=15, k=3, seed=42)
modelo_kan_inicial = KAN(width=[dimension_entrada_mc, 8, num_clases_totales], grid=15, k=3, seed=42)

if os.path.exists(archivo_modelo_multiclase):
    modelo_kan_mc.load_state_dict(torch.load(archivo_modelo_multiclase, weights_only=True))
else:
    if len(dataset_mc['train_input']) > 0:
        modelo_kan_mc.fit(dataset_mc, opt="LBFGS", steps=20, log=10)
        torch.save(modelo_kan_mc.state_dict(), archivo_modelo_multiclase)

predicciones_mc = modelo_kan_mc(X_test_tensor).detach()
clases_reales = torch.argmax(Y_test_tensor, dim=1) if len(Y_test_tensor) > 0 else torch.tensor([])

Y_test_bin = label_binarize(clases_reales.numpy(), classes=list(range(num_clases_totales)))
Y_pred_score = predicciones_mc.numpy()

# Plot 6A 
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
for i in range(num_clases_totales):
    if len(Y_test_bin) > 0 and len(np.unique(Y_test_bin[:, i])) > 1:
        fpr, tpr, _ = roc_curve(Y_test_bin[:, i], Y_pred_score[:, i])
        roc_auc = auc(fpr, tpr)
        label = f'{nombres_clases_dinamicas[i]} (AUC = {roc_auc:.4f})'
    else:
        fpr, tpr = [0, 1], [0, 1]
        label = f'{nombres_clases_dinamicas[i]} (AUC = N/A)'
    ax_roc.plot(fpr, tpr, color=colores_dinamicos[i], lw=2, label=label)

ax_roc.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle='--')
ax_roc.set_xlim([-0.01, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate', fontweight='bold')
ax_roc.set_ylabel('True Positive Rate', fontweight='bold')
ax_roc.set_title('Multiclass ROC Curves (OVR)', fontweight='bold')
ax_roc.grid(True, linestyle='--', linewidth=0.5)
ax_roc.legend(loc="lower right")

for _, spine in ax_roc.spines.items():
    spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(1.2)
fig_roc.savefig(os.path.join(DIR_FIGS, "Fig6_A_Multiclass_ROC.pdf"), bbox_inches='tight')

# Plot 6B 
try:
    coef_abs = torch.abs(modelo_kan_mc.act_fun[0].coef).detach() 
    pesos_matriz = torch.mean(coef_abs, dim=-1) 
    
    if pesos_matriz.shape[0] == dimension_entrada_mc:
        pesos_l1 = torch.mean(pesos_matriz, dim=1).numpy()
    else:
        pesos_l1 = torch.mean(pesos_matriz, dim=0).numpy() 
        
    num_vars_a_graficar = min(15, dimension_entrada_mc)
    top_indices = np.argsort(pesos_l1)[-num_vars_a_graficar:][::-1] 
    idx_max = top_indices[0] 
    
    num_neuronas_ocultas = 8
    idx_entrada_local = idx_max % dimension_entrada_mc
    idx_oculta_local = idx_max % num_neuronas_ocultas
    
    grid_inicial = modelo_kan_inicial.act_fun[0].grid[idx_entrada_local].detach().numpy()
    grid_final = modelo_kan_mc.act_fun[0].grid[idx_entrada_local].detach().numpy()

    fig_kan_int, axes_int = plt.subplots(1, 3, figsize=(18, 5))

    barras = axes_int[0].bar(range(num_vars_a_graficar), pesos_l1[top_indices], color='indigo')
    axes_int[0].set(xlabel=f'Variable Ranking (Top {num_vars_a_graficar})', ylabel='Activation L1 Norm', title='Attribution Spectrum')
    axes_int[0].set_xticks(range(num_vars_a_graficar))
    axes_int[0].set_xticklabels([f"Var_{idx}" for idx in top_indices], rotation=45, ha='right', fontsize=9)

    y_dummy = np.zeros_like(grid_inicial)
    axes_int[1].scatter(grid_inicial, y_dummy + 0.1, color='gray', marker='|', s=200, label='Initial Grid (Uniform)')
    axes_int[1].scatter(grid_final, y_dummy - 0.1, color='red', marker='|', s=200, label='Final Grid (Adapted)')
    axes_int[1].set_ylim(-0.5, 0.5)
    axes_int[1].set_yticks([])
    axes_int[1].set(xlabel='Latent Space (1D Domain)', title='Knot Redistribution (Grid)')
    axes_int[1].legend()

    x_sweep = torch.linspace(-2, 2, 100).unsqueeze(1)
    act_inicial_completa = modelo_kan_inicial.act_fun[0](x_sweep)[0].detach().numpy()
    act_final_completa = modelo_kan_mc.act_fun[0](x_sweep)[0].detach().numpy()
    
    act_inicial = act_inicial_completa[:, idx_oculta_local]
    act_final = act_final_completa[:, idx_oculta_local]

    axes_int[2].plot(x_sweep.numpy(), act_inicial, color='gray', linestyle='--', lw=2, label='Untrained Spline (Iter 0)')
    axes_int[2].plot(x_sweep.numpy(), act_final, color='blue', lw=2, label='Trained Spline (Final Iter)')
    axes_int[2].set(xlabel=r'Input $x$', ylabel=r'Output $\phi(x)$', title='Plastic Spline Deformation')
    axes_int[2].legend()

    plt.tight_layout()
    fig_kan_int.savefig(os.path.join(DIR_FIGS, "Fig6_B_KAN_Interpretability.pdf"), bbox_inches='tight')

except Exception as e:
    print(f"[Warning] Skipped KAN interpretation plots: {e}")

## Step 7: Symbolic Regression
fases = ["W1", "W2", "W3", "W4", "W5"]
dimension_entrada_total = G.N * 15

if os.path.exists(ruta_ecuacion) and os.path.exists(archivo_stats_simbolicas):
    with open(archivo_stats_simbolicas, "rb") as f:
        stats = pickle.load(f)
else:
    try:
        modelo_kan_mc = modelo_kan_mc.prune(node_th=1e-2, edge_th=1e-2)
        modelo_kan_mc.auto_symbolic(lib=['x', 'x^2', 'x^3', 'exp', 'sin', 'abs'])

        variables_simbolicas = []
        for fase in fases:
            for i in range(G.N): variables_simbolicas.append(sympy.Symbol(f"λV_{i}_{fase}"))
            for i in range(G.N): variables_simbolicas.append(sympy.Symbol(f"λf_gen_{i}_{fase}"))
            for i in range(G.N): variables_simbolicas.append(sympy.Symbol(f"λf_load_{i}_{fase}"))

        formulas_multiclase = modelo_kan_mc.symbolic_formula(var=variables_simbolicas)[0]
        formulas_str = [str(f) for f in formulas_multiclase]

        with open(ruta_ecuacion, "w", encoding="utf-8") as f:
            f.write("TOPOLOGICAL CLASSIFICATION SYSTEM (15N)\n" + "="*80 + "\n")
            for i in range(num_clases_totales):
                if i < len(formulas_str):
                    f.write(f"\n[{nombres_clases_dinamicas[i].upper()}]\n{formulas_str[i]}\n" + "-" * 80 + "\n")

        primitivas = {'sin(x)': 0, 'exp(x)': 0, 'x²': 0, 'x³': 0, '|x|': 0}
        matriz_supervivencia = np.zeros((num_clases_totales, 5))
        variables_unicas_sobrevivientes = set()

        for i, form_str in enumerate(formulas_str):
            if i >= num_clases_totales: break
            primitivas['sin(x)'] += form_str.count('sin')
            primitivas['exp(x)'] += form_str.count('exp')
            primitivas['x²'] += form_str.count('**2')
            primitivas['x³'] += form_str.count('**3')
            primitivas['|x|'] += form_str.count('Abs')
            
            vars_en_formula = re.findall(r'λ[A-Za-z_0-9]+', form_str)
            for var in vars_en_formula:
                variables_unicas_sobrevivientes.add(var)
                for j, fase in enumerate(fases):
                    if fase in var:
                        matriz_supervivencia[i, j] += 1

        stats = {
            'inputs_iniciales': dimension_entrada_total,
            'inputs_finales': len(variables_unicas_sobrevivientes),
            'primitivas': primitivas,
            'matriz_supervivencia': matriz_supervivencia
        }
        
        with open(archivo_stats_simbolicas, "wb") as f:
            pickle.dump(stats, f)

    except Exception as e:
        print(f"[Warning] Skipped Symbolic Regression: {e}")
        stats = None

if stats is not None:
    fig7, axes7 = plt.subplots(1, 3, figsize=(18, 5))

    etiquetas_poda = ['Input Variables\n(Full Graph)', 'Surviving Variables\n(Symbolic Boundary)']
    valores_poda = [stats['inputs_iniciales'], stats['inputs_finales']]
    
    axes7[0].grid(axis='y', linestyle='--', linewidth=0.5, zorder=0)
    barras_poda = axes7[0].bar(etiquetas_poda, valores_poda, color=['#7f7f7f', '#2ca02c'], width=0.5, edgecolor='black', linewidth=1.2, zorder=3)
    axes7[0].bar_label(barras_poda, padding=3, fontweight='bold', family='serif')
    axes7[0].set_ylabel(r'Number of Variables ($\lambda$)', fontweight='bold')
    axes7[0].set_title('KAN Pruning Impact', fontweight='bold')
    
    nombres_prim = list(stats['primitivas'].keys())
    conteos_prim = list(stats['primitivas'].values())
    
    axes7[1].grid(axis='y', linestyle='--', linewidth=0.5, zorder=0)
    barras_prim = axes7[1].bar(nombres_prim, conteos_prim, color='#1f77b4', edgecolor='black', linewidth=1.2, zorder=3)
    axes7[1].bar_label(barras_prim, padding=3, fontweight='bold', family='serif')
    axes7[1].set_ylabel('Frequency of Use', fontweight='bold')
    axes7[1].set_title('Chosen Activation Functions (Auto-Symbolic)', fontweight='bold')

    sns.heatmap(stats['matriz_supervivencia'], annot=True, fmt="g", cmap="jet", ax=axes7[2],
                xticklabels=["W1 (1s)", "W2 (5s)", "W3 (10s)", "W4 (15s)", "W5 (20s)"],
                yticklabels=nombres_clases_dinamicas, linewidths=1, linecolor='black', 
                cbar_kws={'label': 'Variable Density'})
    axes7[2].set_xlabel('Observation Window', fontweight='bold')
    axes7[2].set_title('Extracted Analytic Variables per Phase', fontweight='bold')

    for ax in axes7:
        for _, spine in ax.spines.items():
            spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(1.2)

    plt.tight_layout()
    fig7.savefig(os.path.join(DIR_FIGS, "Fig7_Symbolic_Interpretation.pdf"), bbox_inches='tight')