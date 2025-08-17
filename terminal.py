from blessed import Terminal
term = Terminal()

def print_banner() -> str:
    """Print the application banner"""
    try:
        from pyfiglet import Figlet
        f = Figlet(font='slant')
        return f.renderText('Job Applyer')
    except ImportError:
        return "🚀 Job Application Automation Engine\n" + "=" * 50

banner = print_banner()
banner_lines = len(banner.split('\n'))

def clear_screen():
    print(term.home + term.clear)

def clear_screen_preserve_banner(banner_lines):
    print(term.move(banner_lines, 0) + term.clear_eos())

def refresh_display_area(banner_lines):
    print(term.move(banner_lines, 0) + term.clear_eos())