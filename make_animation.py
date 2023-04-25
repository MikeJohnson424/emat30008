import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tempfile
import os
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer
import webbrowser
import threading
import mimetypes
import time
from math import floor, ceil

def view_animation(x, u, grid, t_steps,title):
    fig, ax = plt.subplots()

    y_min = floor(np.min(u))
    y_max = ceil(np.max(u))

    line, = ax.plot(x, u[:, 0])
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(grid.left, grid.right)

    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title(title)

    def animate(i):
        line.set_data((x, u[:, i]))
        return line,

    ani = FuncAnimation(fig, animate, frames=t_steps, interval=50, blit=True)
    plt.close(fig)  # Prevents the display of the static plot

    # Save the animation as a GIF and serve it
    with tempfile.TemporaryDirectory() as tmpdirname:
        gif_path = os.path.join(tmpdirname, 'animation.gif')
        ani.save(gif_path, writer='imagemagick')

        class CustomHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                with open(gif_path, 'rb') as f:
                    content = f.read()

                content_type, _ = mimetypes.guess_type(gif_path)
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', len(content))
                self.end_headers()
                self.wfile.write(content)

        with TCPServer(("", 0), CustomHandler) as httpd:
            port = httpd.socket.getsockname()[1]
            url = f"http://localhost:{port}/animation.gif"
            print(f"Animation available at: {url}")

            # Start the server in a new thread
            server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            server_thread.start()
            webbrowser.open(url)

            # Close the server automatically after 10 seconds
            time.sleep(10)
            httpd.shutdown()

