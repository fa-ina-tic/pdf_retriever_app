c=get_config()

c.NotebookApp.ip='localhost'
c.NotebookApp.open_browser=False
c.NotebookApp.password_required=False
c.NotebookApp.port=8888
c.NotebookApp.iopub_data_rate_limit=1.0e10  
c.NotebookApp.terminado_settings={'shell_command': ['/bin/bash']}