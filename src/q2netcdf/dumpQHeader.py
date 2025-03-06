#! /usr/bin/env python3
#
# Display the Q-file's header record
#
# Nov-2024, Pat Welch, pat@mousebrains.com


def main():
    try:
        import QHeader
    except:
        from q2netcdf import QHeader

    QHeader.main()

if __name__ == "__main__":
    main()
