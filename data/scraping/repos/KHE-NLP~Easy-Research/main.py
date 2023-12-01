import base64
import io
import os
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from cohereGeneration import get_generation, cleanData
from pdfminer.high_level import extract_text

from parsers import get_paragraphs, make_pptx_slides, get_pdf_text


class ContinueServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        #self.wfile.write(b"<form method=\"POST\" action=\"/pdflink\">")
        #self.wfile.write(b"<input type=\"file\" name=\"data\" />")
        self.wfile.write(b"<textarea id=\"urlPDF\" name=\"data\" ></textarea>")
        self.wfile.write(b"<button name=\"submit\" onclick=\"uploadUrl()\">Submit</button>")
        #self.wfile.write(b"</form>")
        self.wfile.write(b"""<script>
        function uploadUrl() {
    var data = $('#urlPDF')[0].value;
    var formData = new FormData();
    formData.append("data", data);

    $.ajax({
       url: "http://127.0.0.1:12345/pdflink",
       type: "POST",
       data: formData,
       processData: false,
       contentType: false,
       success: function(response) {
            resp = atob(response)
            let blob = new Blob([resp], { type: "*/*" });

            let a = document.createElement('a');
            a.href = window.URL.createObjectURL(blob);
            a.download = "flashread.pptx";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.removeObjectURL(a.href);
       },
       error: function(jqXHR, textStatus, errorMessage) {
           console.log(errorMessage);
       }
    });
}
        </script>""")
        self.wfile.write(b"""<script src="https://code.jquery.com/jquery-3.6.1.min.js"
            integrity="sha256-o88AwQnZB+VDvE9tvIXrMQaPlFFSUTR+nldQm1LuPXQ="
            crossorigin="anonymous"></script>""")

    def do_POST(self):
        line = self.path
        print(line)
        if line[:9] == "/continue":
            content_length = int(self.headers['Content-Length'])
            line = self.rfile.read(content_length)
            line = urllib.parse.parse_qs(line)
            print(line)
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            end = list(get_generation([line[b"start"][0].decode()]))
            self.wfile.write(line[b"start"][0] + bytes(end[0].generations[0].text, "utf-8"))
        elif line[:8] == "/pdffile":
            content_length = int(self.headers['Content-Length'])
            line = self.rfile.read(content_length)
            line = urllib.parse.parse_qs(line)
            print(line)
            text = get_paragraphs(extract_text(io.BytesIO(line[b"data"][0])))
            cleanData(text)
            summs = list(get_generation(text, "summary_generator"))
            titles = list(get_generation(summs, "summary_title"))
            ppt = make_pptx_slides([titles[i] + "\n" + summs[i] for i in range(len(titles))])
            self.send_response(200)
            self.send_header("Content-type", "*/*")
            self.end_headers()
            #ppt.save("test.pptx")
            b = bytes()
            ppt.save(io.BytesIO(b))
            print(b)
            self.wfile.write(base64.b64encode(b))
        elif line[:8] == "/pdflink":
            content_length = int(self.headers['Content-Length'])
            line = self.rfile.read(content_length)
            line = line.replace(b"\r", b"").split(b"\n")[3]
            print(line)
            text = get_paragraphs(get_pdf_text(line.decode()))
            cleanData(text)
            summs = list(e.generations[-1].text for e in get_generation(text[1:2], "summary_generator"))
            print("Summaries done")
            titles = list(e.generations[-1].text for e in get_generation(summs, "summary_title"))
            print("titles done")
            ppt = make_pptx_slides([titles[i] + "\n" + summs[i] for i in range(len(titles))])
            self.send_response(200)
            self.send_header("Content-type", "*/*")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "*")
            self.end_headers()
            #ppt.save("test.pptx")
            #ppt.save(self.wfile)
            #filename = os.path.join(application.root_path, 'test.pptx')
            #prs.save(filename)
            #return send_file(filename_or_fp=filename)
            bio = io.BytesIO(bytes())
            ppt.save(bio)
            self.wfile.write(base64.b64encode(bio.getvalue()))
        else:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"BAD")
        return


if __name__ == "__main__":
    server = HTTPServer(("", 12345), ContinueServer)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
