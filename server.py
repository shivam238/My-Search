import http.server
import socketserver

PORT = 5000


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


with socketserver.TCPServer(("0.0.0.0", PORT), NoCacheHandler) as httpd:
    httpd.allow_reuse_address = True
    print(f"Serving on http://0.0.0.0:{PORT}")
    httpd.serve_forever()