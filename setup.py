from setuptools import find_packages, setup

requirements = """
python-dotenv==1.1.1
joblib==1.3.1
py-xbrl==2.2.8
scikit-learn==1.7.0
torchmetrics==1.7.3
tqdm==4.67.1
langchain-openai==0.3.27
langchain-core==0.3.66
matplotlib==3.10.3
peft==0.15.2
pyaml==25.5.0
GitPython==3.1.44
hyperopt==0.2.7
ray[tune]==2.47.1
treelib==1.8.0
seaborn==0.13.2
bitsandbytes==0.46.0
optimum==1.26.1
graphviz==0.21
auto-gptq==0.7.1
trl==0.19.0
torch==2.7.0
tiktoken==0.9.0
ratelimit==2.2.1
backoff==2.2.1
google-genai==1.23.0
pdfplumber==0.11.7
bs4==0.0.2
pypdf2==3.0.1
missingpy==0.2.0
transformers==4.53.0
unsloth==2025.6.9
cut-cross-entropy==25.1.1
xformers==0.0.30
accelerate==1.8.1
unsloth-zoo==2025.6.7
protobuf==3.20.3
sentence-transformers==4.1.0
# vllm==0.9.1
lxt
triton
token-count
pmdarima

#MMTS
gluonts[torch]
chronos-forecasting>=2.0
timesfm
reformer_pytorch
typer_config
"""
setup(
    name="researchpkg",
    author="",
    author_email="",
    description="{description}",
    packages=find_packages(),
    dependency_links=[],
    install_requires=requirements.strip().split("\n"),
    # include_package_data: to install data from MANIFEST.in
    include_package_data=True,
    python_requires=">=3.10.4",
    scripts=[],
    zip_safe=False,
)
