import os
import sys
import shutil

import warnings
from pathlib import Path
from PyPDF2 import PdfFileReader, PdfFileWriter

# Path of the GitHub repository
REPO_DIR = Path(__file__).absolute().parent


def make_report(template_file,
                report_data_file = None,
                report_python_dir = None,
                output_file = None,
                output_dir = None,
                build_dir = None,
                quiet = True,
                clean = True,
                remove_output = False,
                remove_build = False,
                creation_script = None,
                venv_dir = None):

    """
    Create an RMarkdown report

    :param template_file: Path of the RMarkdown template
                          This should be a file ending in .Rmd or .Rnw

    :param report_data_file: Path of a Pickle or Dill file that contains Python data.
                             The data will be loaded into the report

    :param report_python_dir: Working directory for all Python code called in RMarkdown
                              Use this if your RMarkdown report has to import code that you wrote

    :param output_file: Path of the file produced by compiling the template
                        This is typically PDF file but could be any other valid output file type for RMarkdown (e.g., HTML)
                        Set as [template_file].pdf by default

    :param output_dir: Parent directory of output_file
                       By default, this is set to the parent directory of the PDF File (if it exists).
                       Otherwise, it is set as the current working directory

    :param build_dir: Directory where the PDF file will be built
                      Set to pdf_dir by default

    :param quiet: Flag. Set as True to remove messages produced by RMarkdown during compilation

    :param clean: Flag. Set as True to remove intermediate files produced by RMarkdown

    :param remove_output: Flag. Set as True to remove PDF after compilation.
                          Use this if you only want to keep Tex files but not the PDF

    :param remove_build: Flag. Set as True to remove component files to produce PDF
                         Use this if you only want to keep the PDF without the files used to create it

    :param creation_script: Path of the RScript used to create report in R.
                            Set as `make_report.R'

    :param venv_dir: Path of the Python interpreter used by R;
                            Set as the same Python environment that is calling this script by default

    :return: pdf_file: Path of the PDF file produced after compilation
    """

    # check required files
    template_file = Path(template_file)
    assert template_file.exists()
    if template_file.suffix.lower() not in ('.rmd', '.rnw'):
        warnings.warn('''template_file is an unknown file type.  It should either be an RMarkdown file with extension '.Rmd' or a Sweave file with extension .Rnw''')

    if creation_script is None:
        creation_script = REPO_DIR / 'make_report.R'
    else:
        creation_script = Path(creation_script).with_suffix('.R')
        assert creation_script.exists()

    if report_python_dir is None:
        report_python_dir = REPO_DIR
    else:
        report_python_dir = Path(report_python_dir)
        assert report_python_dir.is_dir() and report_python_dir.exists()

    if report_data_file is not None:
        report_data_file = Path(report_data_file)
        assert report_data_file.exists()

    # output file and directory
    if output_file is None:
        output_file = template_file.with_suffix('.pdf')
    else:
        output_file = Path(output_file)

    if output_dir is None:
        output_dir = output_file.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok = True)

    # virtual environment
    if venv_dir is None:
        venv_dir = Path(sys.executable)
    else:
        venv_dir = Path(venv_dir)

    # remove existing files
    if output_file.exists():
        output_file.unlink()

    # build directory
    if build_dir is None:
        build_dir = output_dir / output_file.stem
    else:
        build_dir = Path(build_dir)

    build_dir.mkdir(exist_ok = True)

    # setup build command command
    cmd = ['Rscript "%s" "%s" "%s"' % (creation_script, template_file, report_data_file)]
    cmd.append('--report_python_dir "%s"' % report_python_dir)
    cmd.append('--output_file "%s"' % output_file.name)
    cmd.append('--output_dir "%s"' % output_dir)
    cmd.append('--build_dir "%s"' % build_dir)
    cmd.append('--venv_dir "%s"' % venv_dir)
    #cmd.append('--report_python_env "%s"' % report_python_env)

    if clean:
        cmd.append('--clean')

    if quiet:
        cmd.append('--quiet')

    cmd.append('--run_pandoc')

    # run command
    cmd = ' '.join(cmd)

    out = os.system(cmd)

    if remove_output:
        output_file.unlink()

    if remove_build:
        shutil.rmtree(build_dir, ignore_errors = True)

    return output_file


def open_file(file_name):
    """
    open a file using the System viewer
    :param file_name: path of the file
    :return: None
    """
    f = Path(file_name)
    assert f.exists(), 'file not found: %s' % str(f)
    cmd = 'open "%s"' % str(f)
    os.system(cmd)


def merge_pdfs(pdf_files, merged_file, delete_after_merge = False):
    """
    Merge a list of PDFs into a single merged PDF.
    Merged PDF will only include PDFs that are found on disk.

    :param pdf_files: List of PDF files to merged. List elements must be strings or paths pointing to PDF files. Function will issue a warning if any file does not exist on disk.
    :param merged_file: Name of the merged PDF file. Must be a string or path. Function will issue a warning if a file does not exist on disk.
    :param delete_after_merge: Set to True if you want to delete component PDF files after merge
    :return: merged_file: Path to the merged PDF file
    """
    assert isinstance(pdf_files, (list, set))

    # Cast files to Path objects
    pdf_files = [Path(p) for p in pdf_files]

    if len(pdf_files) == 0:
        warnings.warn('pdf_files does not contain any files')

    # Combine component files into PDF
    pdf_writer = PdfFileWriter()
    for p in pdf_files:
        if p.exists():
            pdf_reader = PdfFileReader(str(p))
            for page in range(pdf_reader.getNumPages()):
                if pdf_reader.getPage(page).getContents():
                    pdf_writer.addPage(pdf_reader.getPage(page))
        else:
            warnings.warn('Could not find file %s on disk' % str(p))

    # Write out the merged PDF
    f = Path(merged_file)

    if f.exists():
        warnings.warn('Overwriting %s' % str(merged_file))
        f.unlink()

    with open(f, 'wb') as out:
        pdf_writer.write(out)

    if delete_after_merge and f.exists():
        for p in pdf_files:
            Path(p).unlink(missing_ok = True)

    return merged_file
