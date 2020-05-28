import setuptools

setuptools.setup(
        name="bfe-py",
        version="0.1",
        author="Nico Garavito",
        author_email="jngaravitoc@email.arizona.edu",
        description="BFE analysis of N-body sims with Python",
        packages=["bfe", "bfe/ios", "bfe/satellites",\
                "bfe/analysis", "bfe/coefficients"]
        )
