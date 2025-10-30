rec {
    pkgs = import <nixpkgs> {};
    tersets = pkgs.stdenv.mkDerivation rec {
        stdenv = pkgs.stdenv;
        pkg-config = pkgs.pkg-config;
        binutils = pkgs.binutils;
        zig = pkgs.zig_0_14;
        buildInputs = [ stdenv pkg-config binutils zig ];
        name = "TerseTS";
        description = "collection of lossy timeseries compression algorithms";
        # builder = ./nix-builder.sh;
        src = ./.;
    };
}
