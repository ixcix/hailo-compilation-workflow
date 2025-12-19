import torch
# import logging
import os
import plotly.graph_objects as go
from typing import Union, Tuple


def nms_3d(box_scores: torch.Tensor, 
           box_preds: torch.Tensor,
           iou_threshold: float = 0.5):
    """
    Perform 3D Non-Maximum Suppression on a set of bounding boxes.
    
    :param box_scores: Tensor de forma (N,) con los scores de las cajas.
    :param box_preds: Tensor de forma (N, 7) con las coordenadas de las cajas en formato (x, y, z, dx, dy, dz, θ).
    :param iou_threshold: Umbral de intersección sobre unión (IoU), entre 0 y 1. Por defecto 0.5.
    
    :return: Tensor con los índices de las cajas seleccionadas después de aplicar NMS y los scores seleccionados.
    """
   
    # Validación de entrada
    if not isinstance(box_scores, torch.Tensor):
        box_scores = torch.tensor(box_scores)
    if not isinstance(box_preds, torch.Tensor):
        box_preds = torch.tensor(box_preds)
    
    if box_scores.ndim != 1 or box_preds.ndim != 2 or box_preds.size(1) != 7:
        raise ValueError("Dimensiones de entrada incorrectas. Se espera (N,) para box_scores y (N,7) para box_preds.")
    
    if not (0 <= iou_threshold <= 1):
        raise ValueError(f"iou_threshold debe estar entre 0 y 1. Recibido: {iou_threshold}")

    selected_indices = []
    selected_scores = []
    best_boxes_hist = []

    # Guardar los índices originales
    original_indices = torch.arange(len(box_scores))
    
    while box_preds.size(0) > 0:
    
        # Ordenar por scores en orden descendente
        sorted_indices = torch.argsort(box_scores, descending=True)
        box_scores = box_scores[sorted_indices]
        box_preds = box_preds[sorted_indices]
        original_indices_sorted = original_indices[sorted_indices]
        
        # Guardar el índice original de la caja de mayor score
        highest_score_idx = original_indices_sorted[0]
        highest_score_box = box_preds[0]
        highest_score_value = box_scores[0]

        # Convertir (x, y, z, dx, dy, dz, θ) a (x_min, y_min, z_min, x_max, y_max, z_max)
        x_min = highest_score_box[0] - highest_score_box[3] / 2
        y_min = highest_score_box[1] - highest_score_box[4] / 2
        z_min = highest_score_box[2] - highest_score_box[5] / 2
        x_max = highest_score_box[0] + highest_score_box[3] / 2
        y_max = highest_score_box[1] + highest_score_box[4] / 2
        z_max = highest_score_box[2] + highest_score_box[5] / 2

        boxes_min = box_preds[:, :3] - box_preds[:, 3:6] / 2
        boxes_max = box_preds[:, :3] + box_preds[:, 3:6] / 2
        
        # Calcular intersecciones
        x_min_inter = torch.max(x_min, boxes_min[:, 0])
        y_min_inter = torch.max(y_min, boxes_min[:, 1])
        z_min_inter = torch.max(z_min, boxes_min[:, 2])
        x_max_inter = torch.min(x_max, boxes_max[:, 0])
        y_max_inter = torch.min(y_max, boxes_max[:, 1])
        z_max_inter = torch.min(z_max, boxes_max[:, 2])
        
        # Volumen de intersección
        inter_vol = torch.clamp(x_max_inter - x_min_inter, min=0) * \
                    torch.clamp(y_max_inter - y_min_inter, min=0) * \
                    torch.clamp(z_max_inter - z_min_inter, min=0)
        
        # Volumen de cada caja
        vol_highest = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        vol_boxes = (boxes_max[:, 0] - boxes_min[:, 0]) * \
                    (boxes_max[:, 1] - boxes_min[:, 1]) * \
                    (boxes_max[:, 2] - boxes_min[:, 2])
        
        # Calcular IoU
        iou_values = inter_vol / (vol_highest + vol_boxes - inter_vol)

        # Crear una máscara con los valores de IoU que superan el umbral
        iou_threshold_mask = iou_values > iou_threshold

        # Filtrar las cajas con IoU > umbral
        iou_threshold_boxes = box_preds[iou_threshold_mask]

        # Guardar la caja con el score más alto
        best_boxes_hist.append(iou_threshold_boxes[0])

        # Eliminar las cajas con un IoU superior al umbral
        box_scores = box_scores[~iou_threshold_mask]
        box_preds = box_preds[~iou_threshold_mask]
        original_indices = original_indices[~iou_threshold_mask]


        # Guardar el índice y score de la caja seleccionada
        selected_indices.append(highest_score_idx.item())
        selected_scores.append(highest_score_value.item())

        #print(f'Indices : {selected_indices}')
        
    # print(f'FIN box preds {box_preds}')
    # print(f'box scores {box_scores}')
    return torch.tensor(selected_indices, dtype=torch.long), torch.tensor(selected_scores)



def plot_3d_boxes(box_preds: torch.Tensor, 
                  box_scores: torch.Tensor, 
                  title: str = "Plot 3D boxes", 
                  save_html: bool = False, 
                  html_filename_path: Union[str, None] = "./plot_3d_boxes.html",
                  color: Tuple[int, int, int] = (255, 0, 0),
                  show_scores: bool = True) -> None:
    """
    Save a 3D plot with the bounding boxes and optionally display each box's score as a label.

    :param box_preds: tensor containing 3D bounding box coordinates with columns 'X', 'Y', 'Z', 'DX', 'DY', 'DZ', 'theta'.
    :param box_scores: tensor containing the scores of the bounding boxes.
    :param title: title of the plot. Default is "Plot 3D boxes".
    :param save_html: whether to save the plot in an HTML file. Default is False.
    :param html_filename_path: name of the HTML file to save. Default is "./plot_3d_boxes.html".
    :param color: RGB color tuple for the boxes. Default is (255, 0, 0) for red.
    :param show_scores: whether to display the scores on the boxes. Default is True.
    """
    
    # Validate the input tensors
    if not isinstance(box_preds, torch.Tensor) or not isinstance(box_scores, torch.Tensor):
        raise TypeError(f"Expected 'box_preds' and 'box_scores' to be torch.Tensors.")
    
    if box_preds.ndim != 2 or box_preds.size(1) != 7:
        raise ValueError(f"'box_preds' should have shape (N, 7) but got {box_preds.shape}.")
    
    if box_scores.ndim != 1 or box_scores.size(0) != box_preds.size(0):
        raise ValueError(f"'box_scores' should have shape (N,) where N is the number of boxes, but got {box_scores.shape}.")

    # Validate HTML filename if saving to HTML
    if save_html:
        if not isinstance(html_filename_path, str) or not html_filename_path.endswith(".html"):
            raise ValueError("Invalid HTML filename. Please provide a valid path with '.html' extension.")

    # Unpack color and set up with desired alpha values
    color_border = f'rgba({color[0]}, {color[1]}, {color[2]}, 1)'
    color_face = f'rgba({color[0]}, {color[1]}, {color[2]}, 0.5)'

    fig = go.Figure()

    for i in range(box_preds.size(0)):  # iterate through rows in the tensor
        box = box_preds[i]
        
        # Unpack box information (center coordinates, dimensions, and rotation)
        x, y, z, dx, dy, dz, theta = box
        
        # Calculate half dimensions for easier use in creating the box
        dx_half, dy_half, dz_half = dx / 2, dy / 2, dz / 2

        # Calculate vertices based on the center and dimensions
        vertices = [
            [x - dx_half, y - dy_half, z - dz_half],  # corner 0
            [x + dx_half, y - dy_half, z - dz_half],  # corner 1
            [x + dx_half, y + dy_half, z - dz_half],  # corner 2
            [x - dx_half, y + dy_half, z - dz_half],  # corner 3
            [x - dx_half, y - dy_half, z + dz_half],  # corner 4
            [x + dx_half, y - dy_half, z + dz_half],  # corner 5
            [x + dx_half, y + dy_half, z + dz_half],  # corner 6
            [x - dx_half, y + dy_half, z + dz_half],  # corner 7
        ]

        # Apply rotation to the vertices (around the z-axis)
        rotation_matrix = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1]
        ])
        
        rotated_vertices = []
        for vertex in vertices:
            vertex_tensor = torch.tensor(vertex, dtype=torch.float32)
            rotated_vertex = torch.matmul(rotation_matrix, vertex_tensor)
            rotated_vertices.append(rotated_vertex.tolist())
        
        # Add trace for the bounding box faces
        rotated_vertices = torch.tensor(rotated_vertices)
        faces = [
            [rotated_vertices[0], rotated_vertices[1], rotated_vertices[2], rotated_vertices[3], rotated_vertices[0]],
            [rotated_vertices[4], rotated_vertices[5], rotated_vertices[6], rotated_vertices[7], rotated_vertices[4]],
            [rotated_vertices[0], rotated_vertices[1], rotated_vertices[5], rotated_vertices[4], rotated_vertices[0]],
            [rotated_vertices[2], rotated_vertices[3], rotated_vertices[7], rotated_vertices[6], rotated_vertices[2]],
            [rotated_vertices[1], rotated_vertices[2], rotated_vertices[6], rotated_vertices[5], rotated_vertices[1]],
            [rotated_vertices[3], rotated_vertices[0], rotated_vertices[4], rotated_vertices[7], rotated_vertices[3]],
        ]

        for face in faces:
            fx, fy, fz = zip(*face)
            fig.add_trace(go.Scatter3d(
                x=fx,
                y=fy,
                z=fz,
                mode='lines',
                line=dict(color=color_border, width=2),
                showlegend=False,
            ))

        # Add filled faces
        triangular_face = [
            [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
            [2, 3, 7], [2, 6, 7], [0, 1, 4], [1, 4, 5],
            [0, 3, 7], [0, 4, 7], [1, 2, 6], [1, 5, 6]
        ]

        fig.add_trace(go.Mesh3d(
            x=rotated_vertices[:, 0].tolist(),
            y=rotated_vertices[:, 1].tolist(),
            z=rotated_vertices[:, 2].tolist(),
            i=[face[0] for face in triangular_face],
            j=[face[1] for face in triangular_face],
            k=[face[2] for face in triangular_face],
            opacity=0.5,
            color=color_face
        ))

        # Optionally add the score label at the centroid
        if show_scores:
            centroid = [(x), (y), (z)]  # Using the center of the box
            fig.add_trace(go.Scatter3d(
                x=[centroid[0]],
                y=[centroid[1]],
                z=[centroid[2]],
                mode='text',
                text=f'{box_scores[i]:.2f}',
                textposition="middle center",
                showlegend=False
            ))

    # Set axis labels and plot title
    fig.update_layout(
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis',
        ),
        title=title,
    )

    # Save as HTML or display
    if save_html:
        html_filename_path = os.path.abspath(html_filename_path)
        fig.write_html(html_filename_path)
        # print(f"Plot saved as HTML: {html_filename_path}")
    else:
        fig.show()