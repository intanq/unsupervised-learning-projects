import open3d as o3d
import os


def load_data(dirname):
    """
    Function to load all ply data with including its labels/espressions
    """
    files = []
    expressions = []
    for expression in os.listdir(dirname):
        if expression.startswith('.'):
            continue
        
        current_expression_folder = os.listdir(f'{dirname}/{expression}')
        for file in current_expression_folder:
            if file.startswith('.'):
                continue
                
            # Load triangle mesh
            currentFilePath = f'{dirname}/{expression}/{file}'
            tm = o3d.io.read_triangle_mesh(currentFilePath)
            tm.compute_vertex_normals()
            tm.compute_triangle_normals()
            tm.paint_uniform_color([0.5,0.5,0.5])
            tm.normalize_normals()
            
            files.append(tm)
            expressions.append(expression)
    return files, expressions