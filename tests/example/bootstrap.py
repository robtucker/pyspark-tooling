import jinja2


_script = jinja2.Template(
    """
#!/bin/sh
set -e -x
# ----------------------------------------------------------------------
#                    Bootstrap EMR cluster
# ---------------------------------------------------------------------- 
sudo python3 --version
sudo python3 -m pip install boto3 awscli findspark numpy pandas pyarrow s3transfer pypandoc
aws s3 cp {{s3_path}}/requirements.txt ./requirements.txt
echo "install requirements from {{gemfury_url}}"
sudo python3 -m pip install -r requirements.txt --extra-index-url {{gemfury_url}}
"""
)


def bootstrap_script(s3_path: str, gemfury_url: str):
    return _script.render(s3_path=s3_path, gemfury_url=gemfury_url)
