{
  description = "Hello world flake using uv2nix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      # hacks = lib.callPackage pyproject-nix.build.hacks { };

      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      pyprojectOverrides = _final: _prev: {
      };

      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      python = pkgs.python312;

      pythonSet =
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.default
              overlay
              pyprojectOverrides
            ]
          );
    in
    {
      packages.x86_64-linux.default = pythonSet.mkVirtualEnv "dev-env" workspace.deps.default;

      devShell.x86_64-linux = pkgs.mkShell {
        packages = with pkgs; [
          python
          black
          uv
        ];

        env = {
          UV_PYTHON_DOWNLOADS = "never";
          UV_PYTHON = python.interpreter;
        }
        // lib.optionalAttrs pkgs.stdenv.isLinux {
          LD_LIBRARY_PATH = lib.makeLibraryPath (
            pkgs.pythonManylinuxPackages.manylinux1
            ++ [
              pkgs.zlib
              pkgs.zstd
              pkgs.libxkbcommon
              pkgs.fontconfig
              pkgs.freetype
              pkgs.dbus
            ]
          );
        };

        C_INCLUDE_PATH = "${pkgs.linuxHeaders}/include";

        shellHook = ''
          unset PYTHONPATH
        '';
      };
    };
}
