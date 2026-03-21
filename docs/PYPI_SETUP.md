# PyPI Publishing Setup Guide

This guide explains how to set up automated publishing to PyPI using GitHub Actions with trusted publishing.

## Overview

The project uses **Trusted Publishing** (OpenID Connect) for secure, token-free PyPI uploads. This is the modern, recommended approach that doesn't require storing API tokens.

## One-Time Setup Steps

### Step 1: Initial Manual Upload (First Time Only)

Before you can configure trusted publishing, the project must exist on PyPI. Do this **once**:

```bash
cd /tmp/q2netcdf  # or wherever you cloned the repo

# Build the package
python3 -m pip install --upgrade build twine
python3 -m build

# Upload to PyPI (you'll be prompted for credentials)
python3 -m twine upload dist/*
```

When prompted:
- **Username**: Your PyPI username
- **Password**: Your PyPI password (or account-wide API token)

This creates the `q2netcdf` project on PyPI.

### Step 2: Configure Trusted Publishing on PyPI

After the first upload:

1. **Go to PyPI project settings**:
   - Visit: https://pypi.org/manage/project/q2netcdf/settings/publishing/

2. **Add a new publisher**:
   - Click "Add a new publisher"
   - Select "GitHub" as the provider

3. **Fill in the details**:
   - **PyPI Project Name**: `q2netcdf`
   - **Owner**: `mousebrains`
   - **Repository name**: `q2netcdf`
   - **Workflow name**: `publish-to-pypi.yml`
   - **Environment name**: `pypi`

4. **Save** the configuration

### Step 3: Create GitHub Environment (Security)

For additional security, create a protected environment in GitHub:

1. Go to your repository settings: https://github.com/mousebrains/q2netcdf/settings/environments
2. Click "New environment"
3. Name it: `pypi`
4. Add protection rules (recommended):
   - ✅ Required reviewers (optional but recommended)
   - ✅ Deployment branch: Only allow `main` branch
5. Save

This prevents accidental publishes from feature branches.

## How It Works

### Automatic Publishing on Release

1. Create a new release on GitHub:
   ```bash
   # Tag and push (already done for v0.4.0)
   git tag -a v0.4.1 -m "Release v0.4.1"
   git push origin v0.4.1
   ```

2. Go to GitHub: https://github.com/mousebrains/q2netcdf/releases/new

3. Select your tag (e.g., `v0.4.1`)

4. Fill in release notes

5. Click "Publish release"

6. **GitHub Actions will automatically**:
   - Build the package
   - Publish to PyPI using trusted publishing
   - Attach the built packages to the GitHub release

### Manual Publishing (Testing)

You can also manually trigger a build without publishing:

1. Go to: https://github.com/mousebrains/q2netcdf/actions/workflows/publish-to-pypi.yml
2. Click "Run workflow"
3. Select branch: `main`
4. **Leave "Publish to PyPI" unchecked** for test build
5. Click "Run workflow"

This builds the package but doesn't publish (useful for testing).

## Workflow Features

The `.github/workflows/publish-to-pypi.yml` workflow:

- ✅ **Builds** the package on every release
- ✅ **Publishes** to PyPI using trusted publishing (no tokens!)
- ✅ **Attaches** built packages to GitHub releases
- ✅ **Environment protection** prevents unauthorized publishes
- ✅ **Manual trigger** option for testing builds

## Security Benefits

Trusted publishing is more secure than API tokens because:

- ✅ No secrets stored in GitHub
- ✅ Short-lived credentials (OIDC tokens)
- ✅ Tied to specific repository and workflow
- ✅ Can't be leaked or stolen
- ✅ Automatically rotated

## Troubleshooting

### "Project does not exist" Error

**Problem**: Trusted publishing configured before first upload

**Solution**: Do the initial manual upload first (Step 1 above)

### "Permission denied" Error

**Problem**: Trusted publishing not configured on PyPI

**Solution**: Complete Step 2 above

### "Environment not found" Error

**Problem**: GitHub environment `pypi` doesn't exist

**Solution**: Create it in GitHub settings (Step 3 above)

### Workflow Doesn't Trigger

**Problem**: Workflow only triggers on published releases

**Solution**: Make sure you "Publish release" not just create a draft

## Version Bump Workflow

When releasing a new version:

```bash
# 1. Update version in pyproject.toml
vim pyproject.toml  # Change version = "0.4.0" to "0.4.1"

# 2. Update CHANGELOG.md
vim documents/CHANGELOG.md  # Add new version section

# 3. Commit changes
git add pyproject.toml documents/CHANGELOG.md
git commit -m "Bump version to 0.4.1"
git push

# 4. Create and push tag
git tag -a v0.4.1 -m "Release v0.4.1"
git push origin v0.4.1

# 5. Create GitHub Release
# Go to: https://github.com/mousebrains/q2netcdf/releases/new
# Select tag v0.4.1, add release notes, click "Publish release"

# 6. GitHub Actions automatically publishes to PyPI!
```

## Checking Build Status

Monitor the build and publish process:

1. **Actions tab**: https://github.com/mousebrains/q2netcdf/actions
2. Look for the "Publish to PyPI" workflow
3. Click to see detailed logs

## Verifying Publication

After publishing, verify on PyPI:

- **Project page**: https://pypi.org/project/q2netcdf/
- **Release history**: https://pypi.org/project/q2netcdf/#history

## Additional Resources

- [PyPI Trusted Publishing Guide](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python Packaging Guide](https://packaging.python.org/)

## Support

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section above
2. Review GitHub Actions logs
3. Check PyPI project settings
4. Consult the PyPI trusted publishing documentation
