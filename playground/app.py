# playground/app.py
from __future__ import annotations
from flask import Flask, request, redirect, url_for, render_template_string
import random
import string

def create_app():
    app = Flask(__name__)

    BASE = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>{{ title }}</title>
      <style>
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; }
        header, footer { padding: 16px 24px; background: #f5f5f5; }
        main { padding: 24px; max-width: 960px; margin: 0 auto; }
        nav a { margin-right: 12px; }
        .btn { display:inline-block; padding:10px 14px; border-radius:8px; border:1px solid #ddd; text-decoration:none; }
        .btn-primary { background:#0057ff; color:white; border-color:#0057ff; }
        .card { padding: 12px 14px; border:1px solid #eee; border-radius:10px; margin:10px 0; }
        .list { list-style: none; padding:0; margin:0; }
        .list li { padding:10px 0; border-bottom:1px solid #eee; }
        .chips { display:flex; gap:8px; flex-wrap: wrap; margin: 8px 0 12px; }
        .chip { background:#eef; padding:6px 10px; border-radius:999px; }
        .banner { position:fixed; bottom:0; left:0; right:0; background:#222; color:#fff; padding:10px 16px; display:flex; justify-content:space-between; align-items:center; }
        .hidden { display:none !important; }
        .field { margin:12px 0; }
        label { display:block; font-weight:600; margin-bottom:6px; }
        input[type="text"], input[type="email"], input[type="password"], select { width:360px; max-width:100%; padding:8px 10px; border:1px solid #ccc; border-radius:8px; }
        .row { display:flex; gap:18px; flex-wrap:wrap; }
      </style>
    </head>
    <body>
      <header>
        <nav aria-label="Main">
          <a class="btn" href="{{ url_for('home') }}">Home</a>
          <a class="btn" href="{{ url_for('list_page') }}">List</a>
          <a class="btn" href="{{ url_for('infinite') }}">Infinite</a>
          <a class="btn" href="{{ url_for('form_page') }}">Form</a>
          <a class="btn" href="{{ url_for('filters') }}">Filters</a>
          <a class="btn" href="{{ url_for('modal_page') }}">Modal</a>
          <a class="btn" href="{{ url_for('upload_page') }}">Upload</a>
          <a class="btn" href="{{ url_for('login_page') }}">Login</a>
          <a class="btn" href="{{ url_for('newtab') }}">New Tab</a>
        </nav>
      </header>
      <main role="main">
        <h1>{{ title }}</h1>
        {{ body|safe }}
      </main>
      <footer><small>Test Playground</small></footer>
    </body>
    </html>
    """

    @app.route("/")
    def home():
        body = """
        <p>Welcome! Use the nav to access test pages.</p>
        <div class="row">
          <a class="btn btn-primary" href="{{ url_for('form_page') }}">Apply Now</a>
          <a class="btn" href="{{ url_for('list_page') }}">Top Stories</a>
          <a class="btn" href="{{ url_for('filters') }}">Browse Filters</a>
          <a class="btn" href="{{ url_for('modal_page') }}">Open Cookie Banner</a>
        </div>
        <p>
          <button class="btn" aria-label="Pricing button">Pricing</button>
        </p>
        """
        return render_template_string(BASE, title="Playground Home", body=body)

    # ---------- LIST ----------
    @app.route("/list")
    def list_page():
        n = int(request.args.get("n", 30))
        prefix = request.args.get("prefix", "Item")
        items = [f"{prefix} {i+1}" for i in range(n)]
        lis = "\n".join([f'<li><a href="{url_for("detail", idx=i+1)}">{label}</a></li>' for i, label in enumerate(items)])
        body = f"""
        <h2>Simple List</h2>
        <p>There are {n} items.</p>
        <ul class="list" role="list">{lis}</ul>
        """
        return render_template_string(BASE, title="List Page", body=body)

    @app.route("/detail/<int:idx>")
    def detail(idx: int):
        body = f"""
        <h2>Detail</h2>
        <p id="detail-text">You opened item {idx}</p>
        <a class="btn" href="{url_for('list_page')}">Back</a>
        """
        return render_template_string(BASE, title=f"Detail {idx}", body=body)

    # ---------- INFINITE (reveal-on-scroll) ----------
    @app.route("/infinite")
    def infinite():
        total = int(request.args.get("total", 200))
        chunk = int(request.args.get("chunk", 30))
        visible = int(request.args.get("visible", chunk))
        items = [f"Result {i+1}" for i in range(total)]
        html_items = []
        for i, label in enumerate(items):
            cls = "" if i < visible else "class='hidden lazy'"
            html_items.append(f"<li {cls}><a href='{url_for('detail', idx=i+1)}'>{label}</a></li>")
        lis = "\n".join(html_items)
        body = f"""
        <p>Scroll down to reveal more results. total={total}, chunk={chunk}</p>
        <ul class="list" id="inf-list" role="feed">{lis}</ul>
        <script>
          (function() {{
            const chunk = {chunk};
            const list = document.getElementById('inf-list');
            function reveal() {{
              const hidden = list.querySelectorAll('li.lazy.hidden');
              for (let i=0; i<Math.min(chunk, hidden.length); i++) hidden[i].classList.remove('hidden');
            }}
            let ticking=false;
            window.addEventListener('scroll', () => {{
              if (ticking) return;
              ticking=true;
              requestAnimationFrame(() => {{
                const nearBottom = (window.innerHeight + window.scrollY) >= (document.body.scrollHeight - 120);
                if (nearBottom) reveal();
                ticking=false;
              }});
            }});
          }})();
        </script>
        """
        return render_template_string(BASE, title="Infinite Results", body=body)

    # ---------- FORM (required + submit -> thank-you) ----------
    @app.route("/form", methods=["GET", "POST"])
    def form_page():
        if request.method == "POST":
            return redirect(url_for("thank_you"))
        body = """
        <form method="POST" action="/form">
          <div class="field">
            <label for="first">First Name *</label>
            <input id="first" name="first" type="text" required aria-required="true" />
          </div>
          <div class="field">
            <label for="last">Last Name *</label>
            <input id="last" name="last" type="text" required aria-required="true" />
          </div>
          <div class="field">
            <label for="email">Email *</label>
            <input id="email" name="email" type="email" required aria-required="true" />
          </div>
          <div class="field">
            <label for="location">Location</label>
            <input id="location" name="location" type="text" placeholder="City" />
          </div>
          <div class="field">
            <label for="role">Role</label>
            <select id="role" name="role">
              <option value="">Select…</option>
              <option>iOS Engineer</option>
              <option>Android Engineer</option>
              <option>Web Engineer</option>
            </select>
          </div>
          <button class="btn btn-primary" type="submit">Submit Application</button>
        </form>
        """
        return render_template_string(BASE, title="Application Form", body=body)

    @app.route("/thank-you")
    def thank_you():
        body = """
        <h2>Thank you</h2>
        <p>Thank you for your application — your application has been submitted.</p>
        """
        return render_template_string(BASE, title="Confirmation", body=body)

    # ---------- MODAL / COOKIE BANNER ----------
    @app.route("/modal")
    def modal_page():
        body = """
        <p>This page shows a cookie banner you can dismiss.</p>
        <div id="cookie-banner" class="banner" role="dialog" aria-label="Cookie banner">
          <span>We use cookies to improve your experience.</span>
          <div>
            <button class="btn" id="decline">Decline</button>
            <button class="btn btn-primary" id="accept">Accept all</button>
          </div>
        </div>
        <script>
          document.getElementById('accept').addEventListener('click',()=>document.getElementById('cookie-banner').remove());
          document.getElementById('decline').addEventListener('click',()=>document.getElementById('cookie-banner').remove());
        </script>
        """
        return render_template_string(BASE, title="Cookie Modal", body=body)

    # ---------- FILTERS with chips ----------
    @app.route("/filters")
    def filters():
        body = """
        <div>
          <h2>Filters</h2>
          <div class="row" role="group" aria-label="Filters">
            <label><input type="checkbox" value="Remote" /> Remote</label>
            <label><input type="checkbox" value="On-site" /> On-site</label>
            <label><input type="checkbox" value="Hybrid" /> Hybrid</label>
            <label><input type="checkbox" value="iOS" /> iOS</label>
            <label><input type="checkbox" value="Android" /> Android</label>
            <label><input type="checkbox" value="Web" /> Web</label>
          </div>
          <div id="chips" class="chips" aria-label="Active filters"></div>
          <ul class="list" role="list">
            <li><a href="/detail/1">iOS Remote position</a></li>
            <li><a href="/detail/2">Android On-site role</a></li>
            <li><a href="/detail/3">Web Hybrid role</a></li>
          </ul>
        </div>
        <script>
          const chips = document.getElementById('chips');
          function renderChips() {
            chips.innerHTML = '';
            document.querySelectorAll('input[type=checkbox]:checked').forEach(cb => {
              const chip = document.createElement('span');
              chip.className='chip';
              chip.textContent = cb.value;
              chips.appendChild(chip);
            });
          }
          document.querySelectorAll('input[type=checkbox]').forEach(cb=>{
            cb.addEventListener('change', renderChips);
          });
        </script>
        """
        return render_template_string(BASE, title="Filters", body=body)

    # ---------- UPLOAD ----------
    @app.route("/upload", methods=["GET", "POST"])
    def upload_page():
        if request.method == "POST":
            return redirect(url_for("thank_you"))
        body = """
        <form method="POST" action="/upload" enctype="multipart/form-data">
          <div class="field">
            <label for="resume">Upload Resume</label>
            <input id="resume" name="resume" type="file" />
          </div>
          <button class="btn btn-primary" type="submit">Submit</button>
        </form>
        """
        return render_template_string(BASE, title="Upload", body=body)

    # ---------- LOGIN ----------
    @app.route("/login")
    def login_page():
        # Simulate client-side login: clicking "Log In" hides form & shows "Log out"
        body = """
        <div id="login-area">
          <form id="login-form">
            <div class="field">
              <label for="email">Email</label>
              <input id="email" type="email" />
            </div>
            <div class="field">
              <label for="password">Password</label>
              <input id="password" type="password" />
            </div>
            <button id="login-btn" class="btn btn-primary" type="button">Log In</button>
          </form>
        </div>
        <div id="post-login" class="hidden">
          <button class="btn">Log out</button>
          <p>Welcome!</p>
        </div>
        <script>
          document.getElementById('login-btn').addEventListener('click', ()=>{
            document.getElementById('login-area').classList.add('hidden');
            document.getElementById('post-login').classList.remove('hidden');
          });
        </script>
        """
        return render_template_string(BASE, title="Login", body=body)

    # ---------- NEW TAB ----------
    @app.route("/newtab")
    def newtab():
        body = """
        <p>Open one of these in a new tab:</p>
        <p>
          <a target="_blank" rel="noopener" href="/detail/101">Docs</a>
          <a target="_blank" rel="noopener" href="/detail/102">Blog</a>
        </p>
        """
        return render_template_string(BASE, title="New Tab", body=body)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5001, debug=True)
